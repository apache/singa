# Communication

---

Different messaging libraries has different benefits and drawbacks. For instance,
MPI provides fast message passing between GPUs (using GPUDirect), but does not
support fault-tolerance well. On the contrary, systems using ZeroMQ can be
fault-tolerant, but does not support GPUDirect. The AllReduce function
of MPI is also missing in ZeroMQ which is efficient for data aggregation for
distributed training. In Singa, we provide general messaging APIs for
communication between threads within a process and across processes, and let
users choose the underlying implementation (MPI or ZeroMQ) that meets their requirements.

Singa's messaging library consists of two components, namely the message, and
the socket to send and receive messages. **Socket** refers to a
Singa defined data structure instead of the Linux Socket.
We will introduce the two components in detail with the following figure as an
example architecture.

<img src="../images/arch/arch2.png" style="width: 550px"/>
<img src="../images/arch/comm.png" style="width: 550px"/>
<p><strong> Fig.1 - Example physical architecture and network connection</strong></p>

Fig.1 shows an example physical architecture and its network connection.
[Section-partition server side ParamShard](architecture.html}) has a detailed description of the
architecture. Each process consists of one main thread running the stub and multiple
background threads running the worker and server tasks. The stub of the main
thread forwards messages among threads . The worker and
server tasks are performed by the background threads.

## Message

<object type="image/svg+xml" style="width: 100px" data="../images/msg.svg" > Not
supported </object>
<p><strong> Fig.2 - Logical message format</strong></p>

Fig.2 shows the logical message format which has two parts, the header and the
content. The message header includes the sender's and receiver's IDs, each consisting of
the group ID and the worker/server ID within the group. The stub forwards
messages by looking up an address table based on the receiver's ID.
There are two sets of messages according to the message type defined below.

  * kGet/kPut/kRequest/kSync for messages about parameters

  * kFeaBlob/kGradBlob for messages about transferring feature and gradient
  blobs of one layer to its neighboring layer

There is a target ID in the header. If the message body is parameters,
the target ID is then the parameter ID. Otherwise the message is related to
layer feature or gradient, and the target ID consists of the layer ID and the
blob ID of that layer. The message content has multiple frames to store the
parameter or feature data.

The API for the base Msg is:

    /**
     * Msg used to transfer Param info (gradient or value), feature blob, etc
     * between workers, stubs and servers.
     *
     * Each msg has a source addr and dest addr identified by a unique integer.
     * It is also associated with a target field (value and version) for ease of
     * getting some meta info (e.g., parameter id) from the msg.
     *
     * Other data is added into the message as frames.
     */
    class Msg {
     public:
      ~Msg();
      Msg();
      /**
       * Construct the msg providing source and destination addr.
       */
      Msg(int src, int dst);
      /**
       * Copy constructor.
       */
      Msg(const Msg& msg);
      /**
       * Swap the src/dst addr
       */
      void SwapAddr();
      /**
       * Add a frame (a chunk of bytes) into the message
       */
      void AddFrame(const void* addr, int nBytes);
      /**
       * @return num of bytes of the current frame.
       */
      int FrameSize();
      /**
       * @return the pointer to the current frame data.
       */
      void* FrameData();
      /**
       * @return the data of the current frame as c string
       */
      char* FrameStr();
      /**
       * Move the cursor to the first frame.
       */
      void FirstFrame();
      /**
       * Move the cursor to the last frame.
       */
      void LastFrame();
      /**
       * Move the cursor to the next frame
       * @return true if the next frame is not NULL; otherwise false
       */
      bool NextFrame();
      /**
       *  Add a 'format' frame to the msg (like CZMQ's zsock_send).
       *
       *  The format is a string that defines the type of each field.
       *  The format can contain any of these characters, each corresponding to
       *  one or two arguments:
       *  i = int (signed)
       *  1 = uint8_t
       *  2 = uint16_t
       *  4 = uint32_t
       *  8 = uint64_t
       *  p = void * (sends the pointer value, only meaningful over inproc)
       *  s = char**
       *
       *  Returns size of the added content.
       */
      int AddFormatFrame(const char *format, ...);
      /**
       *  Parse the current frame added using AddFormatFrame(const char*, ...).
       *
       *  The format is a string that defines the type of each field.
       *  The format can contain any of these characters, each corresponding to
       *  one or two arguments:
       *  i = int (signed)
       *  1 = uint8_t
       *  2 = uint16_t
       *  4 = uint32_t
       *  8 = uint64_t
       *  p = void * (sends the pointer value, only meaningful over inproc)
       *  s = char**
       *
       *  Returns size of the parsed content.
       */
      int ParseFormatFrame(const char* format, ...);

    #ifdef USE_ZMQ
      void ParseFromZmsg(zmsg_t* msg);
      zmsg_t* DumpToZmsg();
    #endif

      /**
       * @return msg size in terms of bytes, ignore meta info.
       */
      int size() const;
      /**
       * Set source addr.
       * @param addr unique identify one worker/server/stub in the current job
       */
      void set_src(int addr) { src_ = addr; }
      /**
       * @return source addr.
       */
      int src() const { return src_; }
      /**
       * Set destination addr.
       * @param addr unique identify one worker/server/stub in the current job
       */
      void set_dst(int addr) { dst_ = addr; }
      /**
       * @return dst addr.
       */
      int dst() const { return dst_; }
      /**
       * Set msg type, e.g., kPut, kGet, kUpdate, kRequest
       */
      void set_type(int type) { type_ = type; }
      /**
       * @return msg type.
       */
      int type() const { return type_; }
      /**
       * Set msg target.
       *
       * One msg has a target to identify some entity in worker/server/stub.
       * The target is associated with a version, e.g., Param version.
       */
      void set_trgt(int val, int version) {
        trgt_val_ = val;
        trgt_version_ = version;
      }
      int trgt_val() const {
        return trgt_val_;
      }
      int trgt_version() const {
        return trgt_version_;
      }

    };

In order for a Msg object to be routed, the source and dest address should be attached.
This is achieved by calling the set_src and set_dst methods of the Msg object.
The address parameter passed to these two methods can be manipulated via a set of
helper functions, shown as below.

    /**
     * Wrapper to generate message address
     * @param grp worker/server group id
     * @param id_or_proc worker/server id or procs id
     * @param type msg type
     */
    inline int Addr(int grp, int id_or_proc, int type) {
      return (grp << 16) | (id_or_proc << 8) | type;
    }

    /**
     * Parse group id from addr.
     *
     * @return group id
     */
    inline int AddrGrp(int addr) {
      return addr >> 16;
    }
    /**
     * Parse worker/server id from addr.
     *
     * @return id
     */
    inline int AddrID(int addr) {
      static const int mask = (1 << 8) - 1;
      return (addr >> 8) & mask;
    }

    /**
     * Parse worker/server procs from addr.
     *
     * @return procs id
     */
    inline int AddrProc(int addr) {
      return AddrID(addr);
    }
    /**
     * Parse msg type from addr
     * @return msg type
     */
    inline int AddrType(int addr) {
      static const int mask = (1 << 8) -1;
      return addr & mask;
    }


## Socket

In SINGA, there are two types of sockets, the Dealer Socket and the Router
Socket, whose names are adapted from ZeroMQ. All connections are of the same type, i.e.,
Dealer<-->Router. The communication between dealers and routers are
asynchronous. In other words, one Dealer
socket can talk with multiple Router sockets, and one Router socket can talk
with multiple Dealer sockets.

### Base Socket

The basic functions of a Singa Socket is to send and receive messages. The APIs
are:

    class SocketInterface {
     public:
      virtual ~SocketInterface() {}
      /**
        * Send a message to connected socket(s), non-blocking. The message
        * will be deallocated after sending, thus should not be used after
        * calling Send();
        *
        * @param msg The message to be sent
        * @return 1 for success queuing the message for sending, 0 for failure
        */
      virtual int Send(Msg** msg) = 0;
      /**
        * Receive a message from any connected socket.
        *
        * @return a message pointer if success; nullptr if failure
        */
      virtual Msg* Receive() = 0;
      /**
       * @return Identifier of the implementation dependent socket. E.g., zsock_t*
       * for ZeroMQ implementation and rank for MPI implementation.
       */
      virtual void* InternalID() const = 0;
    };

A poller class is provided to enable asynchronous communication between routers and dealers.
One can register a set of SocketInterface objects with a poller instance via calling its Add method, and
then call the Wait method of this poll object to wait for the registered SocketInterface objects to be ready
for sending and receiving messages. The APIs of the poller class is shown below.

    class Poller {
     public:
      Poller();
      Poller(SocketInterface* socket);
      /**
        * Add a socket for polling; Multiple sockets can be polled together by
        * adding them into the same poller.
        */
      void Add(SocketInterface* socket);
      /**
        * Poll for all sockets added into this poller.
        * @param timeout Stop after this number of mseconds
        * @return pointer To the socket if it has one message in the receiving
        * queue; nullptr if no message in any sockets,
        */
      SocketInterface* Wait(int duration);

      /**
       * @return true if the poller is terminated due to process interupt
       */
      virtual bool Terminated();
    };


### Dealer Socket

The Dealer socket inherits from the base Socket. In Singa, every Dealer socket
only connects to one Router socket as shown in Fig.1.  The connection is set up
by connecting the Dealer socket to the endpoint of a Router socket.

    class Dealer : public SocketInterface {
     public:
      /*
       * @param id Local dealer ID within a procs if the dealer is from worker or
       * server thread, starts from 1 (0 is used by the router); or the connected
       * remote procs ID for inter-process dealers from the stub thread.
       */
      Dealer();
      explicit Dealer(int id);
      ~Dealer() override;
      /**
        * Setup the connection with the router.
        *
        * @param endpoint Identifier of the router. For intra-process
        * connection, the endpoint follows the format of ZeroMQ, i.e.,
        * starting with "inproc://"; in Singa, since each process has one
        * router, hence we can fix the endpoint to be "inproc://router" for
        * intra-process. For inter-process, the endpoint follows ZeroMQ's
        * format, i.e., IP:port, where IP is the connected process.
        * @return 1 connection sets up successfully; 0 otherwise
        */
      int Connect(const std::string& endpoint);
      int Send(Msg** msg) override;
      Msg* Receive() override;
      void* InternalID() const override;
    };

### Router Socket

The Router socket inherits from the base Socket. One Router socket connects to
at least one Dealer socket. Upon receiving a message, the router forwards it to
the appropriate dealer according to the receiver's ID of this message.

    class Router : public SocketInterface {
     public:
      Router();
      /**
       * There is only one router per procs, hence its local id is 0 and is not set
       * explicitly.
       *
       * @param bufsize Buffer at most this number of messages
       */
      explicit Router(int bufsize);
      ~Router() override;
      /**
       * Setup the connection with dealers.
       *
       * It automatically binds to the endpoint for intra-process communication,
       * i.e., "inproc://router".
       *
       * @param endpoint The identifier for the Dealer socket in other process
       * to connect. It has the format IP:Port, where IP is the host machine.
       * If endpoint is empty, it means that all connections are
       * intra-process connection.
       * @return number of connected dealers.
       */
      int Bind(const std::string& endpoint);
      /**
       * If the destination socket has not connected yet, buffer this the message.
       */
      int Send(Msg** msg) override;
      Msg* Receive() override;
      void* InternalID() const override;

    };

## Implementation

### ZeroMQ

**Why [ZeroMQ](http://zeromq.org/)?** Our previous design used MPI for
communication between Singa processes. But MPI is a poor choice when it comes
to fault-tolerance, because failure at one node brings down the entire MPI
cluster. ZeroMQ, on the other hand, is fault tolerant in the sense that one
node failure does not affect the other nodes. ZeroMQ consists of several basic
communication patterns that can be easily combined to create more complex
network topologies.

<img src="../images/msg-flow.png" style="width: 550px"/>
<p><strong> Fig.3 - Messages flow for ZeroMQ</strong></p>

The communication APIs of Singa are similar to the DEALER-ROUTER pattern of
ZeroMQ. Hence we can easily implement the Dealer socket using ZeroMQ's DEALER
socket, and Router socket using ZeroMQ's ROUTER socket.
The intra-process can be implemented using ZeroMQ's inproc transport, and the
inter-process can be implemented using the tcp transport (To exploit the
Infiniband, we can use the sdp transport). Fig.3 shows the message flow using
ZeroMQ as the underlying implementation. The messages sent from dealers has two
frames for the message header, and one or more frames for the message content.
The messages sent from routers have another frame for the identifier of the
destination dealer.

Besides the DEALER-ROUTER pattern, we may also implement the Dealer socket and
Router socket using other ZeroMQ patterns. To be continued.

### MPI

Since MPI does not provide intra-process communication, we have to implement
it inside the Router and Dealer socket. A simple solution is to allocate one
message queue for each socket. Messages sent to one socket is inserted into the
queue of that socket. We create a SafeQueue class to ensure the consistency of
the queue. All queues are created by the main thread and
passed to all sockets' constructor via *args*.

    /**
     * A thread safe queue class.
     * There would be multiple threads pushing messages into
     * the queue and only one thread reading and popping the queue.
     */
    class SafeQueue{
     public:
      void Push(Msg* msg);
      Msg* Front();
      void Pop();
      bool empty();
    };

For inter-process communication, we serialize the message and call MPI's
send/receive functions to transfer them. All inter-process connections are
setup by MPI at the beginning. Consequently, the Connect and Bind functions do
nothing for both inter-process and intra-process communication.

MPI's AllReduce function is efficient for data aggregation in distributed
training. For example, [DeepImage of Baidu](http://arxiv.org/abs/1501.02876)
uses AllReduce to aggregate the updates of parameter from all workers. It has
similar architecture as [Fig.2](architecture.html),
where every process has a server group and is connected with all other processes.
Hence, we can implement DeepImage in Singa by simply using MPI's AllReduce function for
inter-process communication.
