---
title:      "在 nano-rocksdb 中实现 Prefix 优化：从 Prefix Bloom 到 Prefix Iterator"
date:       2026-04-24 12:00:00
header-img: img/bg-little-universe.jpg
tags:
    - nano-rocksdb
    - leveled
---

这篇文章记录我在 `nano-rocksdb` 中加入 prefix 优化的实现思路。

这里的目标不是完整复刻 RocksDB 的 prefix seek、partitioned filter、hash index 或 memtable prefix bloom，而是做一个适合学习的最小版本：

- 用户可以通过 `Options::prefix_extractor` 告诉 DB 如何提取 key 前缀。
- SSTable 的 bloom filter 可以基于 prefix 构建，而不是基于完整 user key。
- iterator 可以通过 `ReadOptions::prefix_same_as_start` 限制在 `Seek()` 起点的同一前缀内扫描。
- 不做 memtable prefix bloom。
- 不引入新的 table format，不改 compaction 文件选择逻辑。

换句话说，这个实现关注的是 RocksDB prefix 优化里最核心、也最容易在 LevelDB 结构上落地的两件事：

```text
prefix bloom:
  用前缀过滤不相关 data block

prefix iterator:
  用 Seek 起点的前缀约束扫描边界
```

它不是生产级 RocksDB 的完整实现，但足够把 prefix 优化的读路径思想跑通。

---

## 一、为什么 LevelDB 原本没有 prefix 语义

LevelDB 的核心抽象是有序 key-value。

对 LevelDB 来说，key 就是一段完整的 user key。SSTable 中的 key 按 comparator 排序，filter policy 也默认基于完整 user key 工作。

例如有一批 key：

```text
user001:name
user001:email
user001:order
user002:name
user002:email
```

从应用角度看，它们很明显有一个前缀分组：

```text
user001
user002
```

但 LevelDB 并不知道这个语义。它只知道这些是完整 key。

当你执行：

```cpp
iter->Seek("user001:");
```

LevelDB 会从 `"user001:"` 附近开始顺序扫描。至于什么时候离开 `user001` 这个前缀，需要上层自己判断：

```cpp
for (iter->Seek("user001:");
     iter->Valid() && iter->key().starts_with("user001:");
     iter->Next()) {
  ...
}
```

同样，LevelDB 的 bloom filter 也不会天然理解 prefix。它通常回答的是：

```text
这个完整 key 是否可能在这个 data block 中？
```

而不是：

```text
这个 prefix 是否可能在这个 data block 中？
```

RocksDB 的 prefix 优化，本质上就是把“前缀是有意义的”这件事告诉存储引擎，然后让 filter、seek 和 iterator 都能利用这个信息。

---

## 二、这次实现的边界

在动手前，我先给这个学习版实现划了一个边界。

要做：

- `SliceTransform`
- `NewFixedPrefixTransform(n)`
- `Options::prefix_extractor`
- block-based table 的 prefix bloom
- `ReadOptions::prefix_same_as_start`
- prefix iterator 边界包装
- 对 flush、reopen、普通 iterator 行为的测试

不做：

- memtable prefix bloom
- RocksDB 的 full filter
- RocksDB 的 partitioned filter
- prefix hash index
- plain table
- 根据 prefix 裁剪 Version/File picker
- 把 prefix extractor 持久化进 MANIFEST 并做 reopen 兼容校验

这个取舍很重要。

如果一开始就追 RocksDB 的完整能力，很容易陷入 table format、block cache、version set、file metadata、compaction picker 的大量细节。对于学习来说，反而不容易看清主线。

当前版本的主线更简单：

```text
用户提供 prefix_extractor
        |
        v
构建 filter 时写入 prefix
        |
        v
查询 filter 时也查询 prefix
        |
        v
iterator 可选限制在 Seek 起点 prefix 内
```

---

## 三、Prefix Extractor：把前缀语义暴露给 DB

第一步是新增一个 `SliceTransform`。

接口在 `include/leveldb/slice_transform.h`：

```cpp
class LEVELDB_EXPORT SliceTransform {
 public:
  virtual ~SliceTransform();

  virtual const char* Name() const = 0;
  virtual Slice Transform(const Slice& key) const = 0;
};
```

它的含义很直接：

```text
输入完整 user key
输出该 key 的 prefix
```

这次只提供了一个最基础的固定长度前缀提取器：

```cpp
LEVELDB_EXPORT const SliceTransform* NewFixedPrefixTransform(size_t prefix_len);
```

实现位于 `util/slice_transform.cc`：

```cpp
class FixedPrefixTransform : public SliceTransform {
 public:
  explicit FixedPrefixTransform(size_t prefix_len)
      : prefix_len_(prefix_len),
        name_("leveldb.FixedPrefix." + std::to_string(prefix_len)) {}

  const char* Name() const override { return name_.c_str(); }

  Slice Transform(const Slice& key) const override {
    return Slice(key.data(), std::min(prefix_len_, key.size()));
  }

 private:
  const size_t prefix_len_;
  const std::string name_;
};
```

如果配置：

```cpp
options.prefix_extractor = NewFixedPrefixTransform(4);
```

那么：

```text
Transform("user001:name") => "user"
Transform("abc")          => "abc"
Transform("")             => ""
```

这里有一个小细节：如果 key 短于 prefix 长度，直接返回整个 key。这样可以避免 fixed prefix 对短 key 产生越界访问，也让行为更容易理解。

---

## 四、Options：prefix_extractor 是 DB 级配置

`prefix_extractor` 被放在 `Options` 里：

```cpp
const SliceTransform* prefix_extractor = nullptr;
```

这意味着它是 DB 级配置，而不是某一次读请求的临时参数。

原因也很简单：SSTable filter 是写入磁盘的。一个 table 在构建时到底是用完整 key 建 bloom，还是用 prefix 建 bloom，会影响后续所有读取。

典型使用方式是：

```cpp
leveldb::Options options;
options.create_if_missing = true;
options.filter_policy = leveldb::NewBloomFilterPolicy(10);
options.prefix_extractor = leveldb::NewFixedPrefixTransform(4);

leveldb::DB* db = nullptr;
leveldb::Status s = leveldb::DB::Open(options, "/tmp/prefix-demo", &db);
```

默认值是 `nullptr`：

```text
nullptr 表示不开启 prefix 优化，行为回到普通 LevelDB。
```

这也是这个实现保持兼容的关键。

---

## 五、Prefix Bloom：最小侵入地复用 InternalFilterPolicy

LevelDB 原本的 filter 路径已经很清楚：

```text
TableBuilder
  -> FilterBlockBuilder
      -> FilterPolicy::CreateFilter()

Table::InternalGet()
  -> FilterBlockReader::KeyMayMatch()
      -> FilterPolicy::KeyMayMatch()
```

问题在于，LevelDB 的 table 中保存的是 internal key：

```text
user_key + sequence + value_type
```

所以 LevelDB 原本就有一层 `InternalFilterPolicy`，负责把 internal key 转成 user key，再交给用户提供的 `FilterPolicy`。

原始逻辑大致是：

```cpp
mkey[i] = ExtractUserKey(keys[i]);
user_policy_->CreateFilter(keys, n, dst);
```

这次 prefix bloom 的实现就放在这层 wrapper 里。

现在构建 filter 时：

```cpp
Slice* mkey = const_cast<Slice*>(keys);
for (int i = 0; i < n; i++) {
  mkey[i] = ExtractUserKey(keys[i]);
  if (prefix_extractor_ != nullptr) {
    mkey[i] = prefix_extractor_->Transform(mkey[i]);
  }
}
user_policy_->CreateFilter(keys, n, dst);
```

如果没有配置 `prefix_extractor`：

```text
internal key -> user key -> bloom
```

如果配置了 `prefix_extractor`：

```text
internal key -> user key -> prefix -> bloom
```

读 filter 时也必须做完全相同的转换：

```cpp
Slice user_key = ExtractUserKey(key);
if (prefix_extractor_ != nullptr) {
  user_key = prefix_extractor_->Transform(user_key);
}
return user_policy_->KeyMayMatch(user_key, f);
```

这个对称性非常重要。

filter 构建时写入的是 prefix，查询时也必须查询 prefix。否则就会出现：

```text
写入 bloom: prefix("aa1") = "aa"
查询 bloom: full_key("aa1") = "aa1"
```

这会导致本来存在的数据被误判成不存在。Bloom filter 允许 false positive，但不能允许 false negative。

---

## 六、为什么 filter policy name 要变化

还有一个容易忽略的细节：filter policy 的名字。

LevelDB 的 filter meta block 名字里会包含 filter policy name。读取 table 时，也会用这个名字找对应 filter。

如果完整 key bloom 和 prefix bloom 使用完全相同的名字，就有兼容风险：

```text
同一个 bloom policy name
但 filter 内容的 key 空间已经变了
```

所以这里把 `InternalFilterPolicy::Name()` 改成：

```cpp
name_(p == nullptr ? ""
      : prefix_extractor == nullptr
          ? p->Name()
          : std::string(p->Name()) + "." + prefix_extractor->Name())
```

如果没有 prefix：

```text
leveldb.BuiltinBloomFilter2
```

如果使用固定 4 字节 prefix：

```text
leveldb.BuiltinBloomFilter2.leveldb.FixedPrefix.4
```

这不是完整的 MANIFEST 级兼容校验，但至少让 table filter 的 meta block 名字和实际编码方式绑定在一起。

对于学习版实现来说，这是一个性价比很高的保护。

---

## 七、DBImpl 构造时如何把 prefix_extractor 传进去

`Options::prefix_extractor` 默认是 `nullptr`，真正的对象由用户设置。

传递路径是：

```text
用户设置 options.prefix_extractor
        |
        v
DB::Open(options, ...)
        |
        v
DBImpl::DBImpl(raw_options, dbname)
        |
        +--> internal_filter_policy_(raw_options.filter_policy,
                                     raw_options.prefix_extractor)
        |
        +--> options_ = SanitizeOptions(..., raw_options)
```

也就是说，它有两种用途：

第一，传给 `InternalFilterPolicy`，用于 SSTable filter 的构建和查询。

第二，保存在 `options_` 里，后面 iterator 包装时会用到：

```cpp
if (options.prefix_same_as_start && options_.prefix_extractor != nullptr) {
  result = new PrefixConstrainedIterator(result, options_.prefix_extractor);
}
```

这里的 `options` 是 `ReadOptions`，表示本次 iterator 是否启用 prefix 边界。

这里的 `options_` 是 DB 打开时保存下来的 `Options`，里面有用户配置的 `prefix_extractor`。

---

## 八、Prefix Iterator：只包一层，不改底层 iterator

prefix bloom 解决的是 Get 路径上 data block 过滤的问题。

但 prefix scan 还有另一个需求：

```text
从某个 key 开始，只扫描同一 prefix 下的 key。
```

RocksDB 里有 `prefix_same_as_start` 这样的读选项。它的意思可以理解为：

> 用户承诺本次迭代只关心 Seek 起点的同一个 prefix。

这次实现也在 `ReadOptions` 里加了：

```cpp
bool prefix_same_as_start = false;
```

实现上没有去改 `DBIter`、`RangeAwareDBIter`、two level iterator 或 merger iterator，而是在 `DBImpl::NewIterator()` 最后包一层：

```cpp
if (options.prefix_same_as_start && options_.prefix_extractor != nullptr) {
  result = new PrefixConstrainedIterator(result, options_.prefix_extractor);
}
```

`PrefixConstrainedIterator` 的逻辑很小：

```cpp
void Seek(const Slice& target) override {
  Slice prefix = prefix_extractor_->Transform(target);
  prefix_.assign(prefix.data(), prefix.size());
  prefix_active_ = true;
  iter_->Seek(target);
}
```

`Seek()` 时记录起点 prefix。

然后 `Valid()` 时判断当前 key 是否还在这个 prefix 内：

```cpp
bool Valid() const override {
  return iter_->Valid() && (!prefix_active_ || PrefixMatches());
}
```

匹配逻辑是：

```cpp
Slice key_prefix = prefix_extractor_->Transform(iter_->key());
return key_prefix.size() == prefix_.size() &&
       std::memcmp(key_prefix.data(), prefix_.data(), prefix_.size()) == 0;
```

这样用户就可以写：

```cpp
leveldb::ReadOptions read_options;
read_options.prefix_same_as_start = true;

std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(read_options));
for (iter->Seek("aa");
     iter->Valid();
     iter->Next()) {
  ...
}
```

只要 iterator 走到非 `aa` 前缀，`Valid()` 就会返回 false。

---

## 九、为什么 SeekToFirst 和 SeekToLast 不启用 prefix 约束

这个实现里，`prefix_same_as_start` 只约束 `Seek(target)` 之后的扫描。

`SeekToFirst()` 和 `SeekToLast()` 会清掉 prefix 约束：

```cpp
void SeekToFirst() override {
  prefix_active_ = false;
  iter_->SeekToFirst();
}

void SeekToLast() override {
  prefix_active_ = false;
  iter_->SeekToLast();
}
```

原因是：

```text
Seek(target) 有明确 target，所以可以提取 start prefix。
SeekToFirst() 和 SeekToLast() 没有 target，也就没有 start prefix。
```

如果强行给它们套 prefix 语义，反而会让 API 行为变得不清楚。

所以当前语义是：

```text
Seek(target):
  启用 target prefix 边界

SeekToFirst():
  普通全库起始扫描

SeekToLast():
  普通全库末尾扫描
```

这是一个保守设计，也更符合学习版实现的目标。

---

## 十、Next 和 Prev 都受 prefix 边界影响

`PrefixConstrainedIterator` 的 `Next()` 和 `Prev()` 本身不主动跳过或裁剪，只是调用底层 iterator：

```cpp
void Next() override {
  assert(Valid());
  iter_->Next();
}

void Prev() override {
  assert(Valid());
  iter_->Prev();
}
```

真正的边界判断仍然集中在 `Valid()`。

例如 key 顺序是：

```text
a1
a2
b1
b2
```

配置固定 1 字节 prefix 后：

```cpp
iter->Seek("a1");
```

扫描结果是：

```text
a1
a2
invalid
```

因为从 `a2` 再 `Next()` 会走到 `b1`，此时 `Valid()` 发现 prefix 已经从 `a` 变成 `b`，于是返回 false。

反向也是一样：

```cpp
iter->Seek("b2");
```

然后 `Prev()`：

```text
b2
b1
invalid
```

再往前虽然底层 iterator 可能已经到 `a2`，但 wrapper 会把它视为无效，因为它离开了 `b` 前缀。

---

## 十一、和 RocksDB 的关系

这个实现借鉴了 RocksDB prefix 优化的两个核心思想：

第一，prefix 是用户告诉存储引擎的 key 结构信息。

```cpp
options.prefix_extractor = NewFixedPrefixTransform(4);
```

第二，一旦知道 prefix，读路径可以在两个地方利用它：

```text
filter:
  用 prefix bloom 跳过不相关 block

iterator:
  用 Seek 起点 prefix 限制扫描边界
```

但它和 RocksDB 仍然有明显差距。

RocksDB 里 prefix 优化会牵涉更多系统：

- memtable prefix bloom
- block based table full filter
- partitioned filter
- partitioned index
- prefix hash index
- plain table
- prefix seek mode 下的文件裁剪
- prefix extractor 的持久化兼容检查

而当前实现没有做这些。

所以它更像是：

> 在 LevelDB 的原始结构上，用最小改动实现 prefix 优化的核心读路径。

这样的好处是代码非常容易跟：

```text
Options
  -> InternalFilterPolicy
  -> Table filter
  -> DBImpl::NewIterator
  -> PrefixConstrainedIterator
```

坏处也很明确：

- prefix bloom 仍然是 LevelDB 原有的 block-level filter，不是 RocksDB 的 full filter。
- Get 路径不能直接按 prefix 跳过整个 SSTable。
- memtable 查找没有 prefix bloom。
- iterator 只是语义边界包装，没有减少底层 iterator 合并成本。
- prefix extractor 没有写入 MANIFEST 做 DB 级 reopen 校验。

这些都是后续可以继续扩展的方向。

---

## 十二、测试覆盖

为了避免这个功能只停留在“能编译”，我补了几类测试。

### 1. prefix filter 确实按 prefix 工作

测试位于 `db/dbformat_test.cc`：

```cpp
TEST(FormatTest, InternalFilterPolicyFixedPrefix) {
  StringListFilterPolicy user_policy;
  const SliceTransform* prefix_extractor = NewFixedPrefixTransform(2);
  InternalFilterPolicy policy(&user_policy, prefix_extractor);

  std::string keys[] = {IKey("aa1", 100, kTypeValue),
                        IKey("bb1", 100, kTypeValue)};
  Slice key_slices[] = {keys[0], keys[1]};
  std::string filter;
  policy.CreateFilter(key_slices, 2, &filter);

  ASSERT_TRUE(policy.KeyMayMatch(IKey("aa2", 100, kTypeValue), filter));
  ASSERT_TRUE(policy.KeyMayMatch(IKey("bb2", 100, kTypeValue), filter));
  ASSERT_FALSE(policy.KeyMayMatch(IKey("cc1", 100, kTypeValue), filter));

  delete prefix_extractor;
}
```

这里故意用了一个测试用的 `StringListFilterPolicy`，它没有 bloom 的 false positive，可以精确验证：

```text
aa1 和 aa2 因为共享 prefix "aa"，所以 aa2 命中
cc1 prefix 是 "cc"，所以不命中
```

### 2. 不配置 prefix 时保持完整 key 行为

```cpp
TEST(FormatTest, InternalFilterPolicyWithoutPrefixUsesFullUserKey) {
  StringListFilterPolicy user_policy;
  InternalFilterPolicy policy(&user_policy, nullptr);

  ...

  ASSERT_TRUE(policy.KeyMayMatch(IKey("aa1", 100, kTypeValue), filter));
  ASSERT_FALSE(policy.KeyMayMatch(IKey("aa2", 100, kTypeValue), filter));
}
```

这个测试很重要，因为 `prefix_extractor = nullptr` 是默认路径。它必须和原来的 LevelDB 行为一致。

### 3. 短 key 行为

```cpp
TEST(FormatTest, InternalFilterPolicyFixedPrefixHandlesShortKeys) {
  const SliceTransform* prefix_extractor = NewFixedPrefixTransform(4);

  ...

  ASSERT_TRUE(policy.KeyMayMatch(IKey("ab", 100, kTypeValue), filter));
  ASSERT_FALSE(policy.KeyMayMatch(IKey("abc", 100, kTypeValue), filter));
}
```

这个测试验证 fixed prefix 不会把 `"ab"` 和 `"abc"` 当成同一个 prefix。因为当 prefix 长度是 4 时：

```text
Transform("ab")  => "ab"
Transform("abc") => "abc"
```

它们并不相等。

### 4. prefix iterator 的正向和反向边界

测试位于 `db/db_test.cc`：

```cpp
TEST_F(DBTest, PrefixSameAsStartIterator) {
  ...
  read_options.prefix_same_as_start = true;
  Iterator* iter = db_->NewIterator(read_options);

  iter->Seek("a1");
  ...
  iter->Next();
  ASSERT_FALSE(iter->Valid());

  iter->Seek("b2");
  ...
  iter->Prev();
  ASSERT_FALSE(iter->Valid());
}
```

它验证了：

- `Seek("a1")` 后正向扫描不会越过 `a` 前缀。
- `Seek("b2")` 后反向扫描不会越过 `b` 前缀。
- `SeekToFirst()` 仍然保持普通全库扫描语义。

### 5. flush 和 reopen 后 prefix filter 仍能工作

```cpp
TEST_F(DBTest, PrefixFilterSurvivesFlushAndReopen) {
  const FilterPolicy* filter_policy = NewBloomFilterPolicy(10);
  const SliceTransform* prefix_extractor = NewFixedPrefixTransform(2);

  options.filter_policy = filter_policy;
  options.prefix_extractor = prefix_extractor;

  ...

  ASSERT_LEVELDB_OK(dbfull()->TEST_CompactMemTable());
  Reopen(&options);

  ASSERT_EQ("vaa1", Get("aa1"));
  ASSERT_EQ("vaa2", Get("aa2"));
  ASSERT_EQ("vbb1", Get("bb1"));
  ASSERT_EQ("NOT_FOUND", Get("aa-missing"));
  ASSERT_EQ("NOT_FOUND", Get("cc1"));
}
```

这个测试覆盖的是更真实的路径：

```text
写入 MemTable
        |
        v
flush 成 SSTable
        |
        v
SSTable filter 写入磁盘
        |
        v
reopen DB
        |
        v
Get 走 table filter 查询
```

虽然这里没有直接断言读 IO 次数减少，但它验证了 prefix filter 的编码和读取在落盘之后仍然一致。

---

## 十三、一个完整使用示例

假设 key 的前两个字节是用户分组：

```text
aa1
aa2
bb1
bb2
```

可以这样打开 DB：

```cpp
#include "leveldb/db.h"
#include "leveldb/filter_policy.h"
#include "leveldb/options.h"
#include "leveldb/slice_transform.h"

int main() {
  leveldb::Options options;
  options.create_if_missing = true;
  options.filter_policy = leveldb::NewBloomFilterPolicy(10);
  options.prefix_extractor = leveldb::NewFixedPrefixTransform(2);

  leveldb::DB* db = nullptr;
  leveldb::Status s = leveldb::DB::Open(options, "/tmp/prefix-demo", &db);
  assert(s.ok());

  s = db->Put(leveldb::WriteOptions(), "aa1", "vaa1");
  assert(s.ok());
  s = db->Put(leveldb::WriteOptions(), "aa2", "vaa2");
  assert(s.ok());
  s = db->Put(leveldb::WriteOptions(), "bb1", "vbb1");
  assert(s.ok());

  leveldb::ReadOptions read_options;
  read_options.prefix_same_as_start = true;

  leveldb::Iterator* iter = db->NewIterator(read_options);
  for (iter->Seek("aa"); iter->Valid(); iter->Next()) {
    printf("%s -> %s\n", iter->key().ToString().c_str(),
           iter->value().ToString().c_str());
  }
  delete iter;

  delete db;
  delete options.filter_policy;
  delete options.prefix_extractor;
}
```

输出只会包含 `aa` 前缀下的 key：

```text
aa1 -> vaa1
aa2 -> vaa2
```

因为 iterator 走到 `bb1` 时，`PrefixConstrainedIterator::Valid()` 会返回 false。

---

## 十四、当前实现的价值和局限

这次 prefix 优化的价值在于，它用很少的代码把三个关键概念接起来了：

```text
prefix_extractor:
  描述 key 的前缀结构

prefix bloom:
  让 SSTable filter 利用前缀信息

prefix_same_as_start:
  让 iterator 暴露 prefix scan 语义
```

它很适合用来理解 RocksDB prefix 优化的基础思想。

但它不是完整 RocksDB。

如果要继续往生产级方向走，后面至少还有几个方向：

### 1. MemTable prefix bloom

当前 memtable 查找没有利用 prefix bloom。  
如果某些 workload 中 memtable 很大，或者 prefix miss 很多，可以考虑给 MemTable 增加 prefix bloom。

### 2. SSTable-level full filter

当前复用的是 LevelDB 原有 block-level filter。  
它能减少 data block 读取，但不能像 RocksDB full filter 那样更直接地判断整个 SSTable 是否可能包含某个 prefix。

### 3. Prefix-aware file picker

当前 Get 路径仍然会按 LevelDB 原有 version/file 查找逻辑走。  
即使某个文件完全不包含某个 prefix，也不会在 Version 层提前跳过。

### 4. Prefix extractor 持久化校验

当前 filter meta block name 会带上 prefix extractor name，但 MANIFEST 里没有持久化 prefix extractor。

更严格的实现应该像 comparator 或 merge operator 一样，在 reopen 时校验 prefix extractor 是否一致。否则同一个 DB 先用 2 字节 prefix 写入，再用 4 字节 prefix 打开，就可能出现语义不一致。

### 5. 更强的 prefix iterator 优化

当前 `PrefixConstrainedIterator` 是一个边界 wrapper。  
它能保证用户看到的结果不越过 prefix，但底层 iterator 合并成本并没有明显减少。

RocksDB 更进一步的地方在于：当用户承诺 prefix scan 时，底层可以少打开一些无关文件和 block。

---

## 十五、结论

最后把这次实现总结成几句话。

LevelDB 原本只有完整 key 语义，不知道业务 key 中的前缀结构。  
RocksDB 的 prefix 优化，本质上是让用户把这种结构告诉存储引擎，然后让读路径利用它减少无关访问。

在 `nano-rocksdb` 里，这次实现选择了一个学习友好的最小闭环：

- 用 `SliceTransform` 表达 prefix 提取规则。
- 用 `Options::prefix_extractor` 把规则配置到 DB。
- 在 `InternalFilterPolicy` 里把 bloom filter 的输入从 full user key 改成 prefix。
- 在 `ReadOptions::prefix_same_as_start` 下，用 `PrefixConstrainedIterator` 限制扫描边界。
- 保持默认 `nullptr` 路径兼容原始 LevelDB 行为。

所以它解决的不是“完整复刻 RocksDB prefix 子系统”的问题，而是：

> 如何在 LevelDB 的代码结构中，用最小改动实现 prefix-aware filter 和 prefix scan 语义？

这个问题的答案就是：

```text
在 filter wrapper 层转换 key，
在 DB iterator 外层包装边界，
把 prefix 规则作为 Options 传入并保持默认关闭。
```

这也是我觉得它适合作为学习版 RocksDB 功能的原因：  
核心路径清楚，代码改动克制，同时能从真实读路径中看到 prefix 优化的价值。
