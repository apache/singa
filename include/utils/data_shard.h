#ifndef SINGA_UTILS_DATA_SHARD_H_
#define SINGA_UTILS_DATA_SHARD_H_

#include <google/protobuf/message.h>
#include <fstream>
#include <string>
#include <unordered_set>

namespace singa {

/**
 * Data shard stores training/validation/test tuples.
 * Every worker node should have a training shard (validation/test shard
 * is optional). The shard file for training is
 * singa::Cluster::workspace()/train/shard.dat; The shard file for validation
 * is singa::Cluster::workspace()/train/shard.dat; Similar path for test.
 *
 * shard.dat consists of a set of unordered tuples. Each tuple is
 * encoded as [key_len key record_len val] (key_len and record_len are of type
 * uint32, which indicate the bytes of key and record respectively.
 *
 * When Shard obj is created, it will remove the last key if the record size
 * and key size do not match because the last write of tuple crashed.
 *
 * TODO
 * 1. split one shard into multiple shards.
 * 2. add threading to prefetch and parse records
 *
 */
class DataShard {
 public:
  enum {
    // read only mode used in training
    kRead = 0,
    // write mode used in creating shard (will overwrite previous one)
    kCreate = 1,
    // append mode, e.g. used when previous creating crashes
    kAppend = 2
  };

  /**
   * Init the shard obj.
   *
   * @param folder Shard folder (path excluding shard.dat) on worker node
   * @param mode Shard open mode, Shard::kRead, Shard::kWrite or Shard::kAppend
   * @param bufsize Batch bufsize bytes data for every disk op (read or write),
   * default is 100MB
   */
  DataShard(const std::string& folder, int mode);
  DataShard(const std::string& folder, int mode, int capacity);
  ~DataShard();

  /**
   * read next tuple from the shard.
   *
   * @param key Tuple key
   * @param val Record of type Message
   * @return false if read unsuccess, e.g., the tuple was not inserted
   *         completely.
   */
  bool Next(std::string* key, google::protobuf::Message* val);
  /**
   * read next tuple from the shard.
   *
   * @param key Tuple key
   * @param val Record of type string
   * @return false if read unsuccess, e.g., the tuple was not inserted
   *         completely.
   */
  bool Next(std::string* key, std::string* val);
  /**
   * Append one tuple to the shard.
   *
   * @param key e.g., image path
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const std::string& key, const google::protobuf::Message& tuple);
  /**
   * Append one tuple to the shard.
   *
   * @param key e.g., image path
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const std::string& key, const std::string& tuple);
  /**
   * Move the read pointer to the head of the shard file.
   * Used for repeated reading.
   */
  void SeekToFirst();
  /**
   * Flush buffered data to disk.
   * Used only for kCreate or kAppend.
   */
  void Flush();
  /**
   * Iterate through all tuples to get the num of all tuples.
   *
   * @return num of tuples
   */
  int Count();
  /**
   * @return path to shard file
   */
  inline std::string path() { return path_; }

 protected:
  /**
   * Read the next key and prepare buffer for reading value.
   *
   * @param key
   * @return length (i.e., bytes) of value field.
   */
  int Next(std::string* key);
  /**
   * Setup the disk pointer to the right position for append in case that
   * the pervious write crashes.
   *
   * @param path shard path.
   * @return offset (end pos) of the last success written record.
   */
  int PrepareForAppend(const std::string& path);
  /**
   * Read data from disk if the current data in the buffer is not a full field.
   *
   * @param size size of the next field.
   */
  bool PrepareNextField(int size);

 private:
  char mode_ = 0;
  std::string path_ = "";
  // either ifstream or ofstream
  std::fstream fdat_;
  // to avoid replicated record
  std::unordered_set<std::string> keys_;
  // internal buffer
  char* buf_ = nullptr;
  // offset inside the buf_
  int offset_ = 0;
  // allocated bytes for the buf_
  int capacity_ = 0;
  // bytes in buf_, used in reading
  int bufsize_ = 0;
};

}  // namespace singa

#endif  // SINGA_UTILS_DATA_SHARD_H_
