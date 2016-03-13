#ifndef SINGA_THREADPOOL_H_
#define SINGA_THREADPOOL_H_

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace singa {

/**
 * Stores an arbitrary number of threads and facilitates the
 * queueing and distribution of arbitrary tasks to various
 * threads pooled in the thread pool. In the event that a task
 * throws an exception, if the exception type is derived from
 * std::exception then it will be logged, otherwise ignored, and
 * the worker thread will return to the thread pool. Undefined
 * behaviour occurs whenever an operation on the mutex object
 * throws an exception.
 */
class Threadpool final {
public:
  /** 
   * Create a thread pool with the specified size and initialize 
   * each worker thread. Provides basic exception safety.
   */
  explicit Threadpool( const std::size_t size );

  /** 
   * Destroy a thread pool. Must be called from a thread not managed 
   * by this thread pool.
   */
  ~Threadpool();

  // Enqueue the specified task. Provides basic exception safety.
  template<class Fn, class ...Args>
  auto enqueue( Fn &&fn, Args && ...args )
    ->std::future<typename std::result_of<Fn( Args... )>::type>;

  /**
   * Retrieve the number of threads currently managed by this 
   * thread pool.
   */
  int size() { return this->threads_.size(); }

private:
  // Type of a task.
  using task_t = std::function<void()>;

  // A list of worker threads.
  std::vector<std::thread> threads_;
  // A queue of tasks to be distributed and invoked.
  std::queue<task_t> task_queue_;
  // A mutex for atomically getting and setting this threadpool's fields.
  std::mutex queue_mutex_;
  // Condition used to lock each worker thread until its been notified
  // that a potential task is ready to be assigned to it and invoked.
  std::condition_variable cond_handle_task_;
  // Determine if this thread pool is being destroyed or there is at least 
  // one queued tasks to be distributed.
  std::function<bool()> ready_task_;

  // State of the thread pool.
  bool stop;

  Threadpool( const Threadpool & ) = delete;            // Copy constructor disabled.
  Threadpool( Threadpool && ) = delete;                 // Move constructor disabled.
  Threadpool &operator=( const Threadpool & ) = delete; // Copy assignment operator disabled.
  Threadpool &operator=( Threadpool && ) = delete;      // Move assignment operator disabled.
};

// Constructor
inline Threadpool::Threadpool( const size_t size ) : stop( false ) {
  // Set the target function for the ready task predicate.
  this->ready_task_ = [this] {
      return this->stop || !this->task_queue_.empty();
  };

  // Create the specified number of new worker threads.
  for ( size_t k = 0; k < size; ++k ) {
    this->threads_.emplace_back( [this] {
      // This worker thread's assigned task.
      task_t task;
      // Synchronization state of this worker thread.
      bool synchronize = false;

      // Iterate until this worker thread is to be synchronized.
      do {
          {
            // Block this worker thread until it can accept a task from the
            // thread pool.
            std::unique_lock<std::mutex> lock( this->queue_mutex_ );

            this->cond_handle_task_.wait( lock, this->ready_task_ );

            // Determine if we are destroying this thread pool and the tasks
            // queue is empty. If so, synchronize this worker thread, otherwise
            // dequeue a task and assign it to this worker thread.
            if ( this->stop && this->task_queue_.empty() ) {
              synchronize = true;
            } else {
              task = std::move( this->task_queue_.front() );
              this->task_queue_.pop();
            }
          }

        // Determine if we are not synchronizing this worker thread. If so,
        // invoke this worker thread's assigned task.
        if ( !synchronize ) {
          try {
            task();
          } catch ( std::exception & ) {
          // Log the exception caught by the worker thread's assigned task.
          } catch ( ... ) {}
        }
      } while ( !synchronize );
    });
  }
}

// Destructor
inline Threadpool::~Threadpool() {
  // Change the state of the thread pool to "being destroyed".
  {
    std::lock_guard<std::mutex> lock( this->queue_mutex_ );
    this->stop = true;
  }

  // Notify all of the blocked worker threads of this state change.
  this->cond_handle_task_.notify_all();

  // Block the current thread until all of the worker threads from this thread
  // pool are synchronized.
  for ( auto &thread : this->threads_ ) {
    thread.join();
  }
}

template<class Fn, class ...Args>
inline auto Threadpool::enqueue( Fn &&fn, Args && ...args ) 
  -> std::future<typename std::result_of<Fn( Args... )>::type> {

  using return_type = typename std::result_of<Fn( Args... )>::type;

  // Package the specified task in to a new copyable managed object and
  // preserve each argument's value category.
  auto task = std::make_shared<std::packaged_task<return_type()>>(
    std::bind( std::forward<Fn>( fn ), std::forward<Args>( args )... )
  );

  // Get the promised future return value of the specified task.
  std::future<return_type> retval = task->get_future();

  // Enqueue the specified task.
  {
    std::unique_lock<std::mutex> lock( this->queue_mutex_ );

    // Determine if we are preparing to destroy this thread pool. If so, throw
    // a runtime error exception.
    if ( this->stop ) {
      throw std::runtime_error( "Cannot enqueue a task while this thread pool "
        "is preparing to be destroyed." );
    }

      //this->task_queue_.emplace( [task = std::move( task )]{ (*task)(); } );
      this->task_queue_.emplace( [task](){ (*task)(); } );
  }

  // Notify a worker thread that there is a new task to be invoked.
  this->cond_handle_task_.notify_one();

  return retval;
}

} // namespace singa

#endif  // SINGA_THREADPOOL_H_
