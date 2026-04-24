---
title:      "在 nano-rocksdb 中实现 Merge：从写入到 Get 读取路径"
date:       2026-04-21 12:00:00
header-img: img/bg-little-universe.jpg
tags:
    - nano-rocksdb
    - leveled
---

这篇文章记录我在 nano-rocksdb 中加入 Merge 支持的实现思路。整体语义参考 RocksDB：`Merge` 写入的不是最终 value，而是一个 merge operand；读取时需要把同一个 key 上较新的 operands 和更旧的 base value 合并，最后通过 `MergeOperator::FullMerge()` 得到用户可见的 value。

当前实现重点放在读写路径和格式兼容上，暂时没有做 compaction 阶段的 merge folding。也就是说，compaction 不会主动把一串 merge operands 折叠成一个普通 value；真正的合并发生在读取路径里。

## 背景

LevelDB 原本只有两种主要记录：

```cpp
kTypeValue
kTypeDeletion
```

点查时只要按 internal key 顺序找到同一个 user key 下最新的可见记录即可：

```text
Put(k, v)     -> 返回 v
Delete(k)     -> 返回 NotFound
找不到记录      -> 返回 NotFound
```

但 Merge 不一样。Merge 记录本身不是最终值，而是一个操作数：

```text
Merge(k, op)
```

它的含义是：把 `op` 应用到 key `k` 的旧值上。具体怎么应用由用户提供的 merge operator 决定。

例如默认的字符串追加 merge operator 可以理解为：

```text
Put("k", "hello")
Merge("k", " ")
Merge("k", "world")
Get("k") => "hello world"
```

因此，遇到 `kTypeMerge` 时不能像普通 value 一样直接返回，也不能简单跳过继续找旧 value。正确做法是收集 merge operands，然后和 base value 一起交给 merge operator。

## 写入格式

我给 internal value type 增加了一个类型：

```cpp
enum ValueType {
  kTypeDeletion = 0x0,
  kTypeValue = 0x1,
  kTypeRangeDeletion = 0x2,
  kTypeMerge = 0x3,
};
```

MemTable 里的 entry 格式没有专门为 Merge 新建结构，而是复用原来的 internal key 编码：

```cpp
void MemTable::Add(SequenceNumber s, ValueType type, const Slice& key,
                   const Slice& value) {
  // Format of an entry is concatenation of:
  //  key_size     : varint32 of internal_key.size()
  //  key bytes    : char[internal_key.size()]
  //  tag          : uint64((sequence << 8) | type)
  //  value_size   : varint32 of value.size()
  //  value bytes  : char[value.size()]
  ...
  EncodeFixed64(p, (s << 8) | type);
  ...
  table_.Insert(buf);
}
```

所以 `Merge(k, operand)` 在 MemTable 中就是：

```text
internal key = k + sequence + kTypeMerge
value        = operand
```

这样做的好处是 Merge 可以自然进入已有的 WAL、WriteBatch、MemTable、SSTable 和 iterator 框架，不需要给它单独设计一套持久化格式。

## WriteBatch 支持

WriteBatch 也增加了 Merge 记录：

```cpp
void WriteBatch::Merge(const Slice& key, const Slice& value) {
  WriteBatchInternal::SetCount(this, WriteBatchInternal::Count(this) + 1);
  rep_.push_back(static_cast<char>(kTypeMerge));
  PutLengthPrefixedSlice(&rep_, key);
  PutLengthPrefixedSlice(&rep_, value);
}
```

应用到 MemTable 时，`Merge` 会被转成 `kTypeMerge`：

```cpp
void Merge(const Slice& key, const Slice& value) override {
  mem_->Add(sequence_, kTypeMerge, key, value);
  sequence_++;
}
```

DB 层写入时还会检查是否配置了 merge operator：

```cpp
Status DBImpl::Merge(const WriteOptions& options, const Slice& key,
                     const Slice& value) {
  if (options_.merge_operator == nullptr) {
    return Status::InvalidArgument("merge operator is not configured");
  }
  return DB::Merge(options, key, value);
}
```

这一步很重要。因为没有 merge operator，数据库即使存下了 operand，也无法在读取时解释它。

## MergeOperator

Merge 的核心扩展点是 `MergeOperator`：

```cpp
class MergeOperator {
 public:
  virtual ~MergeOperator();

  virtual bool FullMerge(const Slice& key, const Slice* existing_value,
                         const std::vector<Slice>& operand_list,
                         std::string* new_value) const = 0;

  virtual const char* Name() const = 0;
};
```

其中：

- `key` 是当前 user key。
- `existing_value` 是更旧的 base value；如果没有 base value，则为 `nullptr`。
- `operand_list` 是按从旧到新的顺序传入的 merge operands。
- `new_value` 是输出的最终 value。

我的实现里提供了一个默认字符串追加 operator：

```cpp
class DefaultStringAppendMergeOperator : public MergeOperator {
 public:
  const char* Name() const override { return "leveldb.DefaultStringAppend"; }

  bool FullMerge(const Slice& key, const Slice* existing_value,
                 const std::vector<Slice>& operands,
                 std::string* new_value) const override {
    (void)key;
    new_value->clear();
    if (existing_value != nullptr) {
      new_value->append(existing_value->data(), existing_value->size());
    }
    for (const Slice& operand : operands) {
      new_value->append(operand.data(), operand.size());
    }
    return existing_value != nullptr || !operands.empty();
  }
};
```

比如：

```text
Put("name", "nano")
Merge("name", "-")
Merge("name", "rocksdb")
Get("name") => "nano-rocksdb"
```

## 为什么 MemTable::Get 不能直接处理 Merge

原始点查路径中，`MemTable::Get()` 只负责在当前 MemTable 中找到第一个可见 internal entry：

```cpp
if (entry.type == kTypeValue) {
  value->assign(entry.value.data(), entry.value.size());
  return true;
}
if (entry.type == kTypeMerge) {
  *s = Status::InvalidArgument("merge record requires merge operator");
  return true;
}
```

这里不能简单地遇到 `kTypeMerge` 后继续往下找。原因是 merge operands 可能跨多个层级：

```text
memtable:
  seq 105: Merge(k, "+3")

immutable memtable:
  seq 100: Merge(k, "+2")

sstable:
  seq  90: Put(k, "1")
```

最终结果需要跨 memtable、immutable memtable 和 SSTable 收集 operands，再调用 merge operator。单个 `MemTable::Get()` 只看当前 MemTable，没有足够上下文完成这件事。

所以我没有把 merge 解析塞进 `MemTable::Get()`，而是在 DB iterator 层统一处理。

## 带 Merge 的 Iterator

当前实现没有单独新建一个 `MergeAwareDBIter`，而是复用了支持 range deletion 的 iterator：

```cpp
class RangeAwareDBIter : public Iterator {
  ...
};
```

虽然名字叫 `RangeAwareDBIter`，但它现在同时处理：

- range tombstone
- merge operand

入口函数是：

```cpp
Iterator* NewDBIteratorWithRangeDeletions(DBImpl* db,
                                          const Comparator* user_key_comparator,
                                          Iterator* internal_iter,
                                          SequenceNumber sequence,
                                          uint32_t seed) {
  return new RangeAwareDBIter(db, user_key_comparator, internal_iter, sequence,
                              seed);
}
```

`DBImpl::NewIterator()` 中，只要发现可能存在 merge operand，也会走这个 iterator：

```cpp
if (use_range_deletion_path || use_merge_path) {
  return NewDBIteratorWithRangeDeletions(this, user_comparator(), iter,
                                         sequence, seed);
}
```

真正处理 Merge 的核心逻辑在 `RangeAwareDBIter::ReadNextUserEntry()`：

```cpp
bool ReadNextUserEntry(VisibleEntry* entry) {
  ...
  bool resolved = false;
  std::vector<std::string> merge_operands;

  while (true) {
    if (ikey.sequence <= sequence_) {
      switch (ikey.type) {
        case kTypeValue:
          if (!resolved) {
            if (!merge_operands.empty()) {
              Slice raw_value = internal_iter_->value();
              const std::string existing_value(raw_value.data(),
                                               raw_value.size());
              status_ = ResolveMergeValue(db_->merge_operator(), user_key,
                                          merge_operands, &existing_value,
                                          entry);
            } else {
              Slice raw_value = internal_iter_->value();
              entry->value.assign(raw_value.data(), raw_value.size());
              entry->visible = true;
            }
            resolved = true;
          }
          break;

        case kTypeDeletion:
          if (!resolved) {
            if (!merge_operands.empty()) {
              status_ = ResolveMergeValue(db_->merge_operator(), user_key,
                                          merge_operands, nullptr, entry);
            }
            resolved = true;
          }
          break;

        case kTypeMerge:
          if (!resolved) {
            Slice raw_operand = internal_iter_->value();
            merge_operands.emplace_back(raw_operand.data(),
                                        raw_operand.size());
          }
          break;
      }
    }

    internal_iter_->Next();
    ...
  }

  if (!resolved && !merge_operands.empty()) {
    status_ = ResolveMergeValue(db_->merge_operator(), user_key,
                                merge_operands, nullptr, entry);
    resolved = true;
  }
  ...
}
```

它的策略是：

1. 顺着 internal iterator 扫同一个 user key。
2. 遇到 `kTypeMerge`，先把 operand 放进 `merge_operands`。
3. 遇到 `kTypeValue`，把它作为 base value 调用 `FullMerge()`。
4. 遇到 `kTypeDeletion`，说明没有 base value，用 `nullptr` 调用 `FullMerge()`。
5. 如果扫完整个 user key 也没有 base value/deletion，也用 `nullptr` 做 full merge。

合并函数是：

```cpp
Status ResolveMergeValue(const MergeOperator* merge_operator, const Slice& key,
                         const std::vector<std::string>& operands,
                         const std::string* existing_value,
                         VisibleEntry* entry) {
  bool has_value = false;
  Status status = ApplyFullMerge(merge_operator, key, operands, existing_value,
                                 &entry->value, &has_value);
  entry->visible = status.ok() && has_value;
  return status;
}
```

`ApplyFullMerge()` 会把 operands 调整为从旧到新的顺序，然后调用用户的 `FullMerge()`：

```cpp
for (auto it = operands.rbegin(); it != operands.rend(); ++it) {
  operand_slices.push_back(*it);
}

*has_value = merge_operator->FullMerge(key, existing_value_slice,
                                       operand_slices, new_value);
```

为什么需要反转？因为 internal iterator 的顺序是 sequence 从新到旧：

```text
seq 105: Merge(k, "world")
seq 100: Merge(k, " ")
seq  90: Put(k, "hello")
```

扫描时收集到的是：

```text
["world", " "]
```

但 merge operator 通常希望 operands 是从旧到新：

```text
[" ", "world"]
```

所以调用 `FullMerge()` 前需要反转。

## Get 的实现逻辑

`DBImpl::Get()` 的核心工作是选择读路径。

它一开始先确定本次读取的 snapshot：

```cpp
if (options.snapshot != nullptr) {
  snapshot =
      static_cast<const SnapshotImpl*>(options.snapshot)->sequence_number();
} else {
  snapshot = versions_->LastSequence();
}
```

然后判断当前 DB 中是否可能存在 merge operand 或 range tombstone：

```cpp
use_merge_path = MergeOperandsMayExist();
if (!use_merge_path && !checked_for_merge_operands_) {
  mutex_.Unlock();
  Status detection = DetectExistingMergeOperands();
  mutex_.Lock();
  if (!detection.ok()) {
    return detection;
  }
  use_merge_path = MergeOperandsMayExist();
}

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

这里做 detect 是为了处理“打开已有 DB”的场景。磁盘上的 SST 里可能已经有 merge operand，但新的 `DBImpl` 刚启动时，内存标记还不知道。因此第一次读时需要扫描一次，之后通过 `checked_for_merge_operands_` 避免重复扫描。

如果发现可能存在 merge，就直接走 iterator 路径：

```cpp
if (use_merge_path) {
  Iterator* iter = NewIterator(options);
  iter->Seek(key);
  if (iter->Valid() && user_comparator()->Compare(iter->key(), key) == 0) {
    value->assign(iter->value().data(), iter->value().size());
    Status s = iter->status();
    delete iter;
    return s;
  }
  Status s = iter->status();
  delete iter;
  if (s.ok()) {
    s = Status::NotFound(Slice());
  }
  return s;
}
```

这条路径虽然比普通点查重，但语义正确，因为 iterator 能跨 MemTable 和 SSTable 收集同一个 key 的所有相关 internal entries。

如果不存在 merge，也不存在 range deletion，就走原始 LevelDB 快速点查：

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

查询顺序是：

```text
mutable memtable
-> immutable memtable
-> current Version 中的 SSTable
```

如果存在 range deletion，则在 SSTable 点查命中后，还要额外检查这个 key 是否被更新的 range tombstone 覆盖：

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

所以当前 `Get()` 可以总结成：

```text
如果可能存在 merge:
  走 iterator，完整解析 merge

否则如果不存在 range deletion:
  走原始快速点查

否则:
  先点查，再检查 range tombstone 覆盖关系
```

## 与 RocksDB 的关系

这个实现主要参考 RocksDB 的 Merge 语义：

- Merge 写入 operand，而不是立即生成最终 value。
- 读取时收集 operands。
- 使用用户提供的 merge operator 决定如何合并。
- 没有 base value 时，用 `nullptr` 表示 missing existing value。

但当前实现比 RocksDB 简化很多：

- 没有 partial merge。
- 没有 compaction-time merge folding。
- 没有专门的 merge helper 复杂状态机。
- 只在 iterator 读路径中做 full merge。

这种设计适合 nano-rocksdb 当前的学习目标：先把 Merge 的语义跑通，并尽量少侵入原有 LevelDB 的点查和存储格式。

## 一个完整示例

假设默认 merge operator 是字符串追加：

```cpp
leveldb::Options options;
options.create_if_missing = true;
options.merge_operator = leveldb::DefaultMergeOperator();

leveldb::DB* db = nullptr;
leveldb::Status s = leveldb::DB::Open(options, "/tmp/nano-merge-demo", &db);
assert(s.ok());

s = db->Put(leveldb::WriteOptions(), "k", "hello");
assert(s.ok());

s = db->Merge(leveldb::WriteOptions(), "k", " ");
assert(s.ok());

s = db->Merge(leveldb::WriteOptions(), "k", "world");
assert(s.ok());

std::string value;
s = db->Get(leveldb::ReadOptions(), "k", &value);
assert(s.ok());
assert(value == "hello world");

delete db;
```

内部大概会形成：

```text
seq 3: Merge(k, "world")
seq 2: Merge(k, " ")
seq 1: Put(k, "hello")
```

`Get("k")` 时：

```text
1. 发现 DB 中可能存在 merge operand
2. 走 NewIterator()
3. RangeAwareDBIter 扫描同一个 user key
4. 收集 operands: ["world", " "]
5. 找到 base value: "hello"
6. 反转 operands: [" ", "world"]
7. 调用 FullMerge("k", "hello", [" ", "world"])
8. 返回 "hello world"
```

这就是当前 nano-rocksdb 中 Merge 的完整读写闭环。

