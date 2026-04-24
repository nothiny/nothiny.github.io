---
title:      "在 nano-rocksdb 中实现 DeleteRange：从 Range Tombstone 到读写路径"
date:       2026-04-22 12:00:00
header-img: img/bg-little-universe.jpg
tags:
    - nano-rocksdb
    - leveled
---

这篇文章记录我在 nano-rocksdb 中加入 `DeleteRange` 支持的实现思路。整体语义参考 RocksDB：`DeleteRange(begin, end)` 不会立刻枚举并删除区间内的每个 key，而是写入一条 range tombstone。读取时，如果某个 key 落在 tombstone 覆盖范围内，并且 tombstone 的 sequence 比点记录更新，就把这个 key 视为不存在。

当前实现重点放在把 DeleteRange 的语义跑通：WriteBatch、MemTable、SSTable、Get、Iterator 和 compaction 都能识别 range tombstone。相比 RocksDB，这个实现更简单，没有复杂的 RangeDelAggregator，也没有完整的 fragmentation 体系，但核心思路是一致的。

## 背景

LevelDB 原本的删除是点删除：

```cpp
db->Delete(write_options, "k");
```

它会写入一条 `kTypeDeletion`：

```text
seq 10: Delete(k)
seq  9: Put(k, value)
```

读取时看到较新的 deletion，就返回 `NotFound`。

但如果要删除一个大范围：

```text
[user:000000, user:999999)
```

用点删除就需要为每个 key 写一条 deletion marker。这个代价太高：

- 写入放大很大。
- WAL 和 MemTable 会膨胀。
- 后续 compaction 需要处理大量点 tombstone。

`DeleteRange` 的思路是：只写一条范围墓碑。

```text
DeleteRange("user:000000", "user:999999")
```

语义是删除半开区间：

```text
[begin_key, end_key)
```

也就是包含 `begin_key`，不包含 `end_key`。

## Internal Key 类型

我在 internal value type 中加入了 `kTypeRangeDeletion`：

```cpp
enum ValueType {
  kTypeDeletion = 0x0,
  kTypeValue = 0x1,
  kTypeRangeDeletion = 0x2,
};
```

range tombstone 的编码方式是：

```text
internal key = begin_key + sequence + kTypeRangeDeletion
value        = end_key
```

也就是说，范围删除 `[a, z)` 会被编码成：

```text
key   = InternalKey("a", seq, kTypeRangeDeletion)
value = "z"
```

这个设计和 RocksDB 的 range tombstone 模型接近：start key 放在 internal key 里，end key 放在 value 里。

## WriteBatch 支持

WriteBatch 增加了 `DeleteRange` 记录：

```cpp
void WriteBatch::DeleteRange(const Slice& begin_key, const Slice& end_key) {
  WriteBatchInternal::SetCount(this, WriteBatchInternal::Count(this) + 1);
  rep_.push_back(static_cast<char>(kTagRangeDeletion));
  PutLengthPrefixedSlice(&rep_, begin_key);
  PutLengthPrefixedSlice(&rep_, end_key);
}
```

它的 batch 编码是：

```text
kTagRangeDeletion
begin_key
end_key
```

应用到 MemTable 时，会转换成 `kTypeRangeDeletion`：

```cpp
void DeleteRange(const Slice& begin_key, const Slice& end_key) override {
  if (mem_->UserComparator()->Compare(begin_key, end_key) < 0) {
    mem_->Add(sequence_, kTypeRangeDeletion, begin_key, end_key);
  }
  sequence_++;
}
```

这里有两个细节：

第一，只有 `begin_key < end_key` 时才真正插入 tombstone。空区间不会删除任何 key。

第二，即使是空区间，sequence 仍然会递增。这符合 WriteBatch 的语义：batch 中的每条操作都占用一个 sequence。

DB 层也提供了顶层 API：

```cpp
Status DBImpl::DeleteRange(const WriteOptions& options,
                           const Slice& begin_key,
                           const Slice& end_key) {
  if (user_comparator()->Compare(begin_key, end_key) >= 0) {
    return Status::OK();
  }
  WriteBatch batch;
  batch.DeleteRange(begin_key, end_key);
  return Write(options, &batch);
}
```

## MemTable：为什么要双写

`MemTable::Add()` 遇到 `kTypeRangeDeletion` 时做了两件事。

第一，照常插入 MemTable 的 skiplist：

```cpp
table_.Insert(buf);
```

第二，额外把 tombstone 放进 range tombstone 链表：

```cpp
if (type == kTypeRangeDeletion) {
  RangeTombstoneNode* node = new RangeTombstoneNode{
      key.ToString(), value.ToString(), s,
      range_tombstone_head_.load(std::memory_order_relaxed)};
  range_tombstone_head_.store(node, std::memory_order_release);
  range_tombstone_count_.fetch_add(1, std::memory_order_release);
  has_range_deletions_ = true;
}
```

这看起来有点重复，但其实两个结构的用途不同。

skiplist 中的记录用于保持完整的 internal entry 流。MemTable flush 成 SSTable 时，会通过 `MemTable::NewIterator()` 遍历所有 internal entries。如果 range tombstone 不进入 skiplist，flush 时就会丢失 DeleteRange。

额外的 `RangeTombstoneNode` 链表用于点查。普通 skiplist 是按 internal key 排序的，`DeleteRange("a", "z")` 的 start key 是 `"a"`。当我们 `Get("k")` 时，seek 到的是 `"k"` 附近，不能自然看到 start key 为 `"a"` 的 tombstone。所以 MemTable 需要一个额外结构来回答：

```text
key = "k" 是否被某个 range tombstone 覆盖？
```

所以 MemTable 里是双路径：

```text
skiplist:
  负责持久化和 flush

range_tombstone_head_ / fragment cache:
  负责点查时判断覆盖关系
```

## Range Fragment

如果每次 `Get(key)` 都遍历所有 range tombstones，代价会很高。于是 MemTable 会把 tombstone 转换成 fragments：

```cpp
struct RangeDeletionFragment {
  std::string start_key;
  std::string end_key;
  std::vector<SequenceNumber> sequences;
};
```

构建过程大致是：

1. 收集所有 tombstone 的 start 和 end 边界。
2. 按 user comparator 排序。
3. 在相邻边界之间生成不重叠的 fragment。
4. 每个 fragment 保存覆盖它的 tombstone sequences，按从新到旧排列。

例如有两个 tombstone：

```text
seq 100: DeleteRange(a, h)
seq 110: DeleteRange(d, z)
```

会切成类似：

```text
[a, d): sequences = [100]
[d, h): sequences = [110, 100]
[h, z): sequences = [110]
```

这样 `Get("e")` 时，只需要二分找到 `[d, h)` 这个 fragment，然后取对当前 snapshot 可见的最新 sequence。

对应函数是：

```cpp
SequenceNumber MemTable::MaxCoveringRangeDeletion(const Slice& user_key,
                                                  SequenceNumber snapshot) const
```

它返回覆盖 `user_key` 且对 snapshot 可见的最新 tombstone sequence。如果没有覆盖，就返回 0。

## MemTable::Get 如何判断覆盖

没有 range tombstone 时，MemTable 走原始快速路径：

```cpp
if (!has_range_deletions_) {
  ...
}
```

只需要 seek 到目标 key，判断最新点记录是 value 还是 deletion。

一旦 MemTable 中有 range tombstone，就需要同时考虑点记录和范围记录：

```cpp
SequenceNumber newest_point_seq = 0;
const SequenceNumber newest_range_seq =
    MaxCoveringRangeDeletion(user_key, snapshot);
```

然后扫描同一个 user key 的点记录，找到最新的 point value 或 point deletion：

```cpp
enum PointState { kNoPoint, kPointValue, kPointDeletion };
PointState point_state = kNoPoint;
```

最后比较两类记录的 sequence：

```cpp
case kPointValue:
  if (newest_range_seq > newest_point_seq) {
    *s = Status::NotFound(Slice());
  } else {
    value->assign(point_value);
  }
  return true;
```

语义很直接：

```text
如果 range tombstone 比 point value 更新：
  返回 NotFound

否则：
  返回 point value
```

如果没有点记录，但存在覆盖当前 key 的 range tombstone，也返回 `NotFound`：

```cpp
case kNoPoint:
  if (newest_range_seq != 0) {
    *s = Status::NotFound(Slice());
    return true;
  }
  return false;
```

## SSTable：Range Deletion Meta Block

Flush 到 SSTable 时，range tombstone 不写进普通 data block，而是写入单独的 meta block：

```cpp
const char kRangeDelMetaBlockName[] = "leveldb.range_deletions";
```

`TableBuilder::Add()` 中会识别 range deletion：

```cpp
if (UsesInternalComparator(r->options.comparator) && IsRangeDeletion(key)) {
  AddRangeDeletion(key, value);
  return;
}
```

真正写入的是：

```cpp
void TableBuilder::AddRangeDeletion(const Slice& key, const Slice& value) {
  ...
  r->range_deletion_block.Add(key, value);
}
```

最后在 `Finish()` 里把这个 block 写到文件，并把 block handle 放进 metaindex：

```cpp
if (ok() && !r->range_deletion_block.empty()) {
  WriteBlock(&r->range_deletion_block, &range_deletion_block_handle);
  has_range_deletion_block = ok();
}

if (has_range_deletion_block) {
  std::string handle_encoding;
  range_deletion_block_handle.EncodeTo(&handle_encoding);
  meta_index_block.Add(kRangeDelMetaBlockName, handle_encoding);
}
```

这样普通点数据和 range tombstone 在 SSTable 中物理分离：

```text
data blocks:
  普通 key/value 和点删除

range deletion block:
  range tombstones

metaindex:
  "leveldb.range_deletions" -> range deletion block handle
```

这样做有两个好处：

第一，普通 data block 的读取和 filter 不会被 range tombstone 干扰。

第二，range tombstone 可以被单独读取、解析和 fragment 化。

## Table 打开时解析 Range Tombstone

打开 SSTable 时，如果 metaindex 中存在 `leveldb.range_deletions`，就读取 range deletion block：

```cpp
void Table::ReadRangeDeletions(const Slice& range_deletion_handle_value) {
  ...
  rep_->range_deletion_block = new Block(block);
  ...
  if (ParseRangeTombstones(rep_->range_deletion_block, user_comparator,
                           internal_comparator, &tombstones)) {
    BuildRangeDeletionFragments(tombstones, user_comparator,
                                &rep_->range_deletion_fragments);
  }
}
```

Table 层和 MemTable 类似，也会把 tombstones 切成 fragments。后续点查时可以直接二分：

```cpp
Status Table::MaxCoveringRangeDeletion(const ReadOptions& options,
                                       const Slice& user_key,
                                       uint64_t snapshot,
                                       uint64_t* tombstone_sequence) const
```

它会更新 `*tombstone_sequence` 为覆盖当前 key 的最新 tombstone sequence。

## DBImpl::Get 的 range-aware 路径

`DBImpl::Get()` 会先判断当前 DB 是否可能存在 range tombstone：

```cpp
use_range_deletion_path = RangeDeletionsMayExist();
if (!use_range_deletion_path && !checked_for_range_deletions_) {
  mutex_.Unlock();
  Status detection = DetectExistingRangeDeletions();
  mutex_.Lock();
  if (!detection.ok()) {
    return detection;
  }
  use_range_deletion_path = RangeDeletionsMayExist();
}
```

这里的 detect 是为了处理打开已有 DB 的场景。磁盘上的 SSTable 里可能已经有 range deletion block，但新的 `DBImpl` 刚启动时，内存标记还不知道。

如果确认没有 range tombstone，就走原始 LevelDB 快速点查：

```cpp
LookupKey lkey(key, snapshot);
if (mem->Get(lkey, value, &s)) {
  // Done
} else if (imm != nullptr && imm->Get(lkey, value, &s)) {
  // Done
} else {
  s = current->Get(options, lkey, value, &stats);
}
```

如果可能存在 range tombstone，则仍然先按顺序查：

```text
mutable memtable
-> immutable memtable
-> current Version 中的 SSTable
```

MemTable 自己会处理它内部的 range tombstone。问题主要出现在 SSTable：`Version::Get()` 找到一个 point value 后，这个 value 可能被另一个 SSTable 中更新的 range tombstone 覆盖。

所以 SSTable 命中后，还要额外检查：

```cpp
if (s.ok()) {
  SequenceNumber covering_range_sequence = 0;
  Status range_status = current->MaxCoveringRangeDeletion(
      options, key, lkey.internal_key(), snapshot,
      &covering_range_sequence);
  if (!range_status.ok()) {
    s = range_status;
  } else if (covering_range_sequence > stats.found_sequence) {
    s = Status::NotFound(Slice());
  }
}
```

核心判断是：

```text
covering_range_sequence > point_sequence
```

如果覆盖 key 的 range tombstone 比找到的点记录更新，就返回 `NotFound`。

## Iterator 如何跳过被覆盖的 key

普通 `Get()` 只处理一个 key。Iterator 要处理的是有序扫描，需要持续维护当前扫描位置上还有效的 tombstones。

当前实现里有一个 range-aware iterator：

```cpp
class RangeAwareDBIter : public Iterator {
  ...
  std::vector<RangeTombstone> active_tombstones_;
};
```

向前扫描时，它维护一个 active tombstone 集合：

```cpp
void DropExpiredTombstones(const Slice& user_key) {
  size_t dst = 0;
  for (size_t i = 0; i < active_tombstones_.size(); ++i) {
    if (user_comparator_->Compare(active_tombstones_[i].end_key, user_key) >
        0) {
      active_tombstones_[dst++] = active_tombstones_[i];
    }
  }
  active_tombstones_.resize(dst);
}
```

每次读到新的 user key 时，先移除已经过期的 tombstone。所谓过期，就是：

```text
tombstone.end_key <= current_user_key
```

遇到 `kTypeRangeDeletion` 时，把它加入 active 集合：

```cpp
case kTypeRangeDeletion:
  if (user_comparator_->Compare(ikey.user_key, internal_iter_->value()) < 0) {
    RangeTombstone tombstone;
    tombstone.end_key = internal_iter_->value().ToString();
    tombstone.sequence = ikey.sequence;
    active_tombstones_.push_back(std::move(tombstone));
  }
  break;
```

判断当前 point value 是否可见时，取 active tombstones 中最大的 sequence：

```cpp
SequenceNumber MaxCoveringTombstone() const {
  SequenceNumber max_seq = 0;
  for (const RangeTombstone& tombstone : active_tombstones_) {
    if (tombstone.sequence > max_seq) {
      max_seq = tombstone.sequence;
    }
  }
  return max_seq;
}
```

最后只有当 point value 比覆盖它的 tombstone 更新时，才返回给用户：

```cpp
if (have_point && !point_deleted &&
    point_sequence > MaxCoveringTombstone()) {
  entry->key = std::move(user_key);
  entry->value = std::move(point_value);
  entry->visible = true;
}
```

这和 `Get()` 的判断是一致的：

```text
point_sequence > covering_range_sequence:
  point value 可见

否则:
  被 range tombstone 覆盖
```

## Compaction 中如何处理 Range Tombstone

compaction 需要做两件事：

第一，用 range tombstone 删除被它覆盖、并且对所有活跃 snapshot 都不可见的旧 point keys。

第二，保留仍然需要继续向下层传播的 range tombstone。

实现里用 `ActiveRangeTombstoneSet` 维护当前扫描位置的 active tombstones：

```cpp
class ActiveRangeTombstoneSet {
 public:
  explicit ActiveRangeTombstoneSet(SequenceNumber smallest_snapshot)
      : smallest_snapshot_(smallest_snapshot) {}
  ...
};
```

它只把对所有活跃 snapshot 都可见的 tombstone 放进 `visible_sequences_`：

```cpp
tombstone.visible_to_all_snapshots = sequence <= smallest_snapshot_;
```

这样 compaction 丢弃旧 point key 时不会破坏 snapshot 语义。核心判断是：

```cpp
const SequenceNumber covering_range_sequence =
    active_range_tombstones.NewestVisibleSequence();
if (ikey.type != kTypeRangeDeletion &&
    covering_range_sequence > ikey.sequence) {
  drop = true;
}
```

意思是：如果有一个更新的 range tombstone 覆盖当前 point key，并且这个 tombstone 对所有活跃 snapshot 都可见，那么当前 point key 可以安全丢弃。

对于 range tombstone 本身，compaction 不直接写进普通 data block，而是先 buffer：

```cpp
if (parsed && ikey.type == kTypeRangeDeletion &&
    user_comparator()->Compare(ikey.user_key, input->value()) < 0) {
  BufferCompactionRangeTombstone(compact, ikey.user_key, input->value(),
                                 ikey.sequence);
}
```

输出文件切换时，再统一写入 range deletion block：

```cpp
compact->builder->AddRangeDeletion(start.Encode(), truncated_end);
```

这里有一个重要细节：如果 tombstone 跨过了当前 output file 的边界，需要截断并把剩余部分 carry 到下一个文件。

```cpp
if (next_file_first_key != nullptr &&
    user_comparator()->Compare(truncated_end, *next_file_first_key) > 0) {
  truncated_end = *next_file_first_key;
  split_for_next_file = true;
}
```

例如一个 tombstone 是：

```text
DeleteRange(a, z)
```

当前 output file 只覆盖到 `m`，那就先写：

```text
[a, m)
```

然后把剩余部分带到下一个 output file：

```text
[m, z)
```

这样每个 SSTable 的 range deletion block 和它自己的 key range 更匹配，文件 metadata 的 smallest/largest 也更容易维护。

## 和 RocksDB 的关系

这个实现主要参考 RocksDB 的几个核心思想：

- DeleteRange 写入 range tombstone，而不是展开成大量点删除。
- tombstone 使用 `[start, end)` 半开区间。
- start key 编码在 internal key 中，end key 放在 value 中。
- 读取时比较 point sequence 和 covering tombstone sequence。
- compaction 可以利用 tombstone 丢弃被覆盖的旧 point keys。
- range tombstone 在 SSTable 中使用独立的 meta block。

但为了保持 nano-rocksdb 的实现规模可控，这里做了不少简化：

- 没有完整复刻 RocksDB 的 RangeDelAggregator。
- fragment 构建更直接，主要用于点查和 table 内覆盖判断。
- Iterator 的 active tombstone 维护比较朴素。
- compaction 中只做基础的覆盖删除、shadow 判断和 output file 边界截断。
- 没有引入复杂的 range tombstone compaction strategy。

这些简化让代码更适合作为学习版 RocksDB：能看清 DeleteRange 从写入到读取、再到 compaction 的完整链路，而不会一开始就陷入生产级实现的复杂性。

## 一个完整示例

假设有下面的写入：

```cpp
leveldb::DB* db = nullptr;
leveldb::Options options;
options.create_if_missing = true;

leveldb::Status s = leveldb::DB::Open(options, "/tmp/nano-deleterange-demo",
                                      &db);
assert(s.ok());

s = db->Put(leveldb::WriteOptions(), "a", "va");
assert(s.ok());
s = db->Put(leveldb::WriteOptions(), "b", "vb");
assert(s.ok());
s = db->Put(leveldb::WriteOptions(), "c", "vc");
assert(s.ok());
s = db->Put(leveldb::WriteOptions(), "d", "vd");
assert(s.ok());

s = db->DeleteRange(leveldb::WriteOptions(), "b", "d");
assert(s.ok());

std::string value;

s = db->Get(leveldb::ReadOptions(), "a", &value);
assert(s.ok() && value == "va");

s = db->Get(leveldb::ReadOptions(), "b", &value);
assert(s.IsNotFound());

s = db->Get(leveldb::ReadOptions(), "c", &value);
assert(s.IsNotFound());

s = db->Get(leveldb::ReadOptions(), "d", &value);
assert(s.ok() && value == "vd");

delete db;
```

内部大概是：

```text
seq 5: DeleteRange(b, d)
seq 4: Put(d, vd)
seq 3: Put(c, vc)
seq 2: Put(b, vb)
seq 1: Put(a, va)
```

读取结果是：

```text
Get(a) => va
Get(b) => NotFound
Get(c) => NotFound
Get(d) => vd
```

因为 `DeleteRange(b, d)` 是半开区间：

```text
[b, d)
```

它覆盖 `b` 和 `c`，但不覆盖 `d`。

如果之后又写入一个更新的 point value：

```cpp
db->Put(leveldb::WriteOptions(), "c", "new-c");
```

内部变成：

```text
seq 6: Put(c, new-c)
seq 5: DeleteRange(b, d)
seq 3: Put(c, vc)
```

此时：

```text
Get(c) => new-c
```

因为 point value 的 sequence 6 比 range tombstone 的 sequence 5 更新。

这就是 DeleteRange 的核心语义：它不是物理上立即删除一段 key，而是写入一条带 sequence 的范围墓碑；读取和 compaction 再根据 sequence 和 key range 判断哪些点记录应该被隐藏或丢弃。

