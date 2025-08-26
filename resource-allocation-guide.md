# Resource Allocation Systems: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Fundamental Concepts](#fundamental-concepts)
3. [Core Data Structures](#core-data-structures)
4. [Key Algorithms](#key-algorithms)
5. [Allocation Strategies](#allocation-strategies)
6. [Practical Implementation Approaches](#practical-implementation-approaches)
7. [Performance Optimization](#performance-optimization)
8. [Real-World Examples](#real-world-examples)

## Introduction

Resource allocation is a fundamental problem in computer science where limited resources must be distributed among competing consumers efficiently and fairly. This problem appears in various forms:
- **Operating Systems**: CPU scheduling, memory management, I/O device allocation
- **Cloud Computing**: Virtual machine placement, container orchestration
- **Networks**: Bandwidth allocation, packet routing
- **Databases**: Connection pooling, query optimization
- **Distributed Systems**: Load balancing, task scheduling

## Fundamental Concepts

### 1. Resource Types

**Preemptible Resources**
- Can be taken away from current owner without harm
- Example: CPU time, memory pages
- Allows for dynamic reallocation

**Non-preemptible Resources**
- Cannot be taken away without causing failure
- Example: Printers, tape drives, database locks
- Must wait for voluntary release

### 2. Allocation Properties

**Mutual Exclusion**
- A resource can only be used by one process at a time
- Critical for preventing race conditions

**Hold and Wait**
- Process holding resources can request additional ones
- Can lead to deadlock if not managed properly

**No Preemption**
- Resources cannot be forcibly removed
- Simplifies implementation but reduces flexibility

**Circular Wait**
- Chain of processes waiting for resources held by next in chain
- Primary cause of deadlock

### 3. Performance Metrics

**Throughput**
- Number of allocations completed per unit time
- Measures system efficiency

**Latency**
- Time from request to allocation
- Critical for user experience

**Utilization**
- Percentage of resources in use
- Balance between efficiency and availability

**Fairness**
- Equal opportunity or proportional share
- Prevents starvation

## Core Data Structures

### 1. Bitmap (Bit Vector)

**Structure**: Array of bits where each bit represents resource availability
```
Resources: [R0, R1, R2, R3, R4, R5, R6, R7]
Bitmap:    [1,  0,  0,  1,  1,  0,  1,  0]
           (1 = available, 0 = allocated)
```

**Operations**:
- Check availability: O(1) - Test bit at position
- Allocate: O(1) - Set bit to 0
- Deallocate: O(1) - Set bit to 1
- Find first available: O(n) - Scan for first 1

**Space Complexity**: O(n) bits where n = number of resources

**Example Implementation**:
```python
class BitmapAllocator:
    def __init__(self, size):
        self.size = size
        self.bitmap = (1 << size) - 1  # All bits set to 1
    
    def allocate(self):
        if self.bitmap == 0:
            return -1  # No resources available
        
        # Find first available bit using bit manipulation
        position = (self.bitmap & -self.bitmap).bit_length() - 1
        self.bitmap &= ~(1 << position)  # Clear the bit
        return position
    
    def deallocate(self, position):
        self.bitmap |= (1 << position)  # Set the bit
```

### 2. Free List

**Structure**: Linked list of available resources
```
Free List: [R2] -> [R5] -> [R7] -> [R9] -> NULL
```

**Operations**:
- Allocate: O(1) - Remove from head
- Deallocate: O(1) - Add to head
- Check specific resource: O(n) - Must traverse list

**Space Complexity**: O(k) where k = number of free resources

### 3. Buddy System

**Structure**: Binary tree where each node represents a block size
```
                    [1024KB]
                   /        \
             [512KB]          [512KB]
            /      \         /       \
        [256KB] [256KB]  [256KB]  [256KB]
```

**Algorithm**:
1. Round request to next power of 2
2. Find smallest available block ≥ request
3. Split blocks recursively if needed
4. Coalesce buddy blocks on deallocation

**Example**:
```python
class BuddyAllocator:
    def __init__(self, total_size):
        self.total_size = total_size
        self.free_lists = {2**i: [] for i in range(int(math.log2(total_size)) + 1)}
        self.free_lists[total_size] = [0]  # Initially one large block
        
    def allocate(self, size):
        # Round up to next power of 2
        size = 2**math.ceil(math.log2(size))
        
        # Find smallest block >= size
        for block_size in sorted(self.free_lists.keys()):
            if block_size >= size and self.free_lists[block_size]:
                block = self.free_lists[block_size].pop(0)
                
                # Split if necessary
                while block_size > size:
                    block_size //= 2
                    buddy = block + block_size
                    self.free_lists[block_size].append(buddy)
                
                return block, size
        
        return None  # No suitable block found
```

### 4. Bloom Filter

**Structure**: Probabilistic data structure using multiple hash functions
```
Bloom Filter (m=10 bits, k=3 hash functions):
[0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

To check if resource R is allocated:
- h1(R) = 1, h2(R) = 3, h3(R) = 6
- Check positions 1, 3, 6
- If all are 1, R is probably allocated
- If any is 0, R is definitely not allocated
```

**Properties**:
- False positives possible (says allocated when free)
- No false negatives (never says free when allocated)
- Space efficient: ~1.44 bits per item for 1% false positive rate

**Implementation**:
```python
class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size
    
    def _hash(self, item, seed):
        return hash(str(item) + str(seed)) % self.size
    
    def add(self, item):
        for i in range(self.hash_count):
            position = self._hash(item, i)
            self.bit_array[position] = 1
    
    def contains(self, item):
        for i in range(self.hash_count):
            position = self._hash(item, i)
            if self.bit_array[position] == 0:
                return False
        return True
```

### 5. Cuckoo Filter

**Structure**: Hash table with multiple positions per item
```
Table with 2 hash functions:
Position: [0] [1] [2] [3] [4] [5] [6] [7]
Content:  [A] [ ] [B] [C] [ ] [D] [ ] [E]

Item X can be at h1(X) or h2(X)
If both occupied, evict one and relocate
```

**Advantages over Bloom Filter**:
- Supports deletion
- Better locality (check only 2 positions)
- Lower space overhead for low false positive rates

## Key Algorithms

### 1. First-Fit Algorithm

**Concept**: Allocate first resource that satisfies request

```python
def first_fit(resources, request_size):
    for i, resource in enumerate(resources):
        if resource.size >= request_size and resource.is_free:
            return i
    return -1  # No suitable resource found
```

**Characteristics**:
- Time: O(n) worst case
- Simple to implement
- Can lead to fragmentation
- Good for small number of resources

### 2. Best-Fit Algorithm

**Concept**: Allocate smallest resource that satisfies request

```python
def best_fit(resources, request_size):
    best_index = -1
    best_size = float('inf')
    
    for i, resource in enumerate(resources):
        if resource.is_free and resource.size >= request_size:
            if resource.size < best_size:
                best_size = resource.size
                best_index = i
    
    return best_index
```

**Characteristics**:
- Time: O(n) - Must examine all resources
- Minimizes wasted space
- Can create many small fragments
- Good when resource sizes vary significantly

### 3. Worst-Fit Algorithm

**Concept**: Allocate largest available resource

```python
def worst_fit(resources, request_size):
    worst_index = -1
    worst_size = 0
    
    for i, resource in enumerate(resources):
        if resource.is_free and resource.size >= request_size:
            if resource.size > worst_size:
                worst_size = resource.size
                worst_index = i
    
    return worst_index
```

**Characteristics**:
- Leaves larger remaining fragments
- Better for future large requests
- Reduces small unusable fragments

### 4. Round-Robin Algorithm

**Concept**: Allocate resources in circular order

```python
class RoundRobinAllocator:
    def __init__(self, num_resources):
        self.num_resources = num_resources
        self.current = 0
        self.available = [True] * num_resources
    
    def allocate(self):
        start = self.current
        while True:
            if self.available[self.current]:
                self.available[self.current] = False
                allocated = self.current
                self.current = (self.current + 1) % self.num_resources
                return allocated
            
            self.current = (self.current + 1) % self.num_resources
            if self.current == start:
                return -1  # No resources available
```

**Characteristics**:
- Fair distribution
- Prevents starvation
- Simple state management
- Good for homogeneous resources

### 5. Priority-Based Algorithm

**Concept**: Allocate based on request priority

```python
import heapq

class PriorityAllocator:
    def __init__(self):
        self.request_queue = []  # Min heap
        self.available_resources = set()
    
    def request_resource(self, requester_id, priority):
        # Lower number = higher priority
        heapq.heappush(self.request_queue, (priority, requester_id))
    
    def release_resource(self, resource_id):
        self.available_resources.add(resource_id)
        self._try_allocation()
    
    def _try_allocation(self):
        while self.request_queue and self.available_resources:
            priority, requester_id = heapq.heappop(self.request_queue)
            resource_id = self.available_resources.pop()
            return (requester_id, resource_id)
        return None
```

### 6. Banker's Algorithm (Deadlock Avoidance)

**Concept**: Ensure system remains in safe state

```python
class BankersAlgorithm:
    def __init__(self, total_resources):
        self.total_resources = total_resources
        self.available = total_resources.copy()
        self.allocation = {}  # Current allocations
        self.max_demand = {}  # Maximum demand per process
    
    def is_safe_state(self):
        work = self.available.copy()
        finish = {p: False for p in self.allocation}
        
        while True:
            found = False
            for process in self.allocation:
                if not finish[process]:
                    need = self.max_demand[process] - self.allocation[process]
                    if all(need[i] <= work[i] for i in range(len(work))):
                        work = [work[i] + self.allocation[process][i] 
                               for i in range(len(work))]
                        finish[process] = True
                        found = True
            
            if not found:
                break
        
        return all(finish.values())
    
    def request_resources(self, process, request):
        # Check if request exceeds maximum claim
        if any(request[i] > self.max_demand[process][i] 
               for i in range(len(request))):
            return False
        
        # Check if resources available
        if any(request[i] > self.available[i] 
               for i in range(len(request))):
            return False
        
        # Tentatively allocate
        self.available = [self.available[i] - request[i] 
                         for i in range(len(request))]
        self.allocation[process] = [self.allocation[process][i] + request[i] 
                                   for i in range(len(request))]
        
        # Check if still safe
        if self.is_safe_state():
            return True
        else:
            # Rollback
            self.available = [self.available[i] + request[i] 
                            for i in range(len(request))]
            self.allocation[process] = [self.allocation[process][i] - request[i] 
                                      for i in range(len(request))]
            return False
```

## Allocation Strategies

### 1. Static Allocation

**Fixed Partitioning**
- Resources divided into fixed-size partitions
- Each requester gets dedicated partition
- Simple but inflexible
- Can lead to internal fragmentation

**Example**: Memory pages in early operating systems
```
Memory Layout:
[OS: 0-10MB] [Part1: 10-30MB] [Part2: 30-60MB] [Part3: 60-100MB]
```

### 2. Dynamic Allocation

**Variable Partitioning**
- Partitions created on demand
- Size matches request exactly
- No internal fragmentation
- External fragmentation possible

**Compaction Strategy**:
```python
def compact_memory(memory_blocks):
    # Move all allocated blocks to one end
    allocated = []
    free_size = 0
    
    for block in memory_blocks:
        if block.allocated:
            allocated.append(block)
        else:
            free_size += block.size
    
    # Reorganize memory
    current_address = 0
    for block in allocated:
        block.start_address = current_address
        current_address += block.size
    
    # Create single large free block
    return allocated + [FreeBlock(current_address, free_size)]
```

### 3. Hierarchical Allocation

**Multi-Level Resource Management**
```
Global Level: Total system resources
    |
Group Level: Department/tenant quotas
    |
User Level: Individual allocations
```

**Implementation**:
```python
class HierarchicalAllocator:
    def __init__(self, total_resources):
        self.total = total_resources
        self.group_quotas = {}
        self.user_allocations = {}
    
    def set_group_quota(self, group, quota):
        if sum(self.group_quotas.values()) + quota <= self.total:
            self.group_quotas[group] = quota
            return True
        return False
    
    def allocate_to_user(self, user, group, amount):
        group_used = sum(alloc for u, alloc in self.user_allocations.items() 
                        if self.get_user_group(u) == group)
        
        if group_used + amount <= self.group_quotas.get(group, 0):
            self.user_allocations[user] = self.user_allocations.get(user, 0) + amount
            return True
        return False
```

### 4. Token Bucket Algorithm

**Rate-Limited Allocation**
```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
    
    def consume(self, tokens):
        self.refill()
        
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False
    
    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
```

### 5. Weighted Fair Queuing

**Proportional Share Allocation**
```python
class WeightedFairQueue:
    def __init__(self):
        self.queues = {}  # queue_id -> (weight, items)
        self.virtual_time = 0
    
    def enqueue(self, queue_id, item, weight):
        if queue_id not in self.queues:
            self.queues[queue_id] = (weight, [])
        self.queues[queue_id][1].append(item)
    
    def dequeue(self):
        best_queue = None
        best_finish_time = float('inf')
        
        for queue_id, (weight, items) in self.queues.items():
            if items:
                # Virtual finish time = virtual_start_time + size/weight
                finish_time = self.virtual_time + 1.0/weight
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_queue = queue_id
        
        if best_queue:
            self.virtual_time = best_finish_time
            return self.queues[best_queue][1].pop(0)
        return None
```

## Practical Implementation Approaches

### 1. Deterministic Allocation with Minimal State

**Concept**: Use cryptographic techniques to generate unique, deterministic sequences

```python
import hashlib

class DeterministicAllocator:
    def __init__(self, total_resources, secret_key):
        self.total_resources = total_resources
        self.secret_key = secret_key
        self.user_cursors = {}  # user_id -> position in sequence
    
    def generate_sequence(self, user_id):
        """Generate deterministic permutation for user"""
        sequence = []
        seen = set()
        
        for i in range(self.total_resources):
            # Generate pseudo-random but deterministic value
            seed = f"{self.secret_key}:{user_id}:{i}".encode()
            hash_val = int(hashlib.sha256(seed).hexdigest(), 16)
            
            # Map to unused resource
            resource_id = hash_val % self.total_resources
            attempts = 0
            
            while resource_id in seen and attempts < self.total_resources:
                attempts += 1
                resource_id = (resource_id + 1) % self.total_resources
            
            if attempts < self.total_resources:
                seen.add(resource_id)
                sequence.append(resource_id)
        
        return sequence
    
    def allocate(self, user_id, count):
        cursor = self.user_cursors.get(user_id, 0)
        sequence = self.generate_sequence(user_id)
        
        if cursor + count > len(sequence):
            return []  # No more resources
        
        allocated = sequence[cursor:cursor + count]
        self.user_cursors[user_id] = cursor + count
        
        return allocated
```

**Advantages**:
- Storage: O(users) - Only store cursor position
- Reproducible: Same sequence every time
- No coordination: Each server can compute independently

### 2. Bucket-Based Allocation

**Concept**: Divide resources into buckets for efficient tracking

```python
class BucketAllocator:
    def __init__(self, total_resources, bucket_size):
        self.num_buckets = (total_resources + bucket_size - 1) // bucket_size
        self.bucket_size = bucket_size
        self.user_buckets = {}  # user_id -> set of exhausted buckets
        self.bucket_states = {}  # (user_id, bucket_id) -> allocation bitmap
    
    def allocate(self, user_id, count):
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = set()
        
        allocated = []
        exhausted_buckets = self.user_buckets[user_id]
        
        # Try each bucket
        for bucket_id in range(self.num_buckets):
            if bucket_id in exhausted_buckets:
                continue
            
            # Get or create bucket state
            key = (user_id, bucket_id)
            if key not in self.bucket_states:
                self.bucket_states[key] = 0  # Bitmap of allocated resources
            
            bitmap = self.bucket_states[key]
            bucket_allocated = []
            
            # Find free resources in bucket
            for i in range(self.bucket_size):
                if not (bitmap & (1 << i)):  # Resource i is free
                    resource_id = bucket_id * self.bucket_size + i
                    bucket_allocated.append(resource_id)
                    bitmap |= (1 << i)  # Mark as allocated
                    
                    if len(allocated) + len(bucket_allocated) >= count:
                        break
            
            self.bucket_states[key] = bitmap
            allocated.extend(bucket_allocated)
            
            # Mark bucket as exhausted if full
            if bitmap == (1 << self.bucket_size) - 1:
                exhausted_buckets.add(bucket_id)
            
            if len(allocated) >= count:
                break
        
        return allocated[:count]
```

### 3. Hybrid Approach with Multiple Strategies

**Concept**: Combine different algorithms based on system state

```python
class HybridAllocator:
    def __init__(self, total_resources):
        self.total_resources = total_resources
        
        # Different allocators for different scenarios
        self.fast_allocator = BitmapAllocator(total_resources)
        self.space_efficient = DeterministicAllocator(total_resources, "secret")
        self.fair_allocator = RoundRobinAllocator(total_resources)
        
        # Metrics for choosing strategy
        self.allocation_count = 0
        self.contention_level = 0
        
    def allocate(self, user_id, count, priority=0):
        self.allocation_count += 1
        
        # Choose strategy based on current conditions
        if self.contention_level < 0.3:
            # Low contention - use fast bitmap
            return self.fast_allocator.allocate(count)
        elif self.contention_level < 0.7:
            # Medium contention - use fair round-robin
            return self.fair_allocator.allocate(count)
        else:
            # High contention - use space-efficient deterministic
            return self.space_efficient.allocate(user_id, count)
    
    def update_contention(self, failed_attempts, successful_attempts):
        total = failed_attempts + successful_attempts
        if total > 0:
            self.contention_level = failed_attempts / total
```

## Performance Optimization

### 1. Caching Strategies

**Multi-Level Cache**
```python
class CachedAllocator:
    def __init__(self, base_allocator):
        self.base = base_allocator
        self.l1_cache = {}  # In-memory, per-user
        self.l2_cache = {}  # Shared Redis cache
        self.cache_ttl = 60  # seconds
        
    def allocate(self, user_id, count):
        # Check L1 cache
        if user_id in self.l1_cache:
            cached = self.l1_cache[user_id]
            if len(cached) >= count:
                return [cached.pop() for _ in range(count)]
        
        # Check L2 cache
        l2_key = f"cache:{user_id}"
        cached = self.redis_client.spop(l2_key, count)
        if len(cached) == count:
            return list(cached)
        
        # Fallback to base allocator
        allocated = self.base.allocate(user_id, count)
        
        # Prefetch extra for cache
        extra = self.base.allocate(user_id, count * 2)
        if extra:
            self.l1_cache[user_id] = extra[:count]
            self.redis_client.sadd(l2_key, *extra[count:])
            self.redis_client.expire(l2_key, self.cache_ttl)
        
        return allocated
```

### 2. Lock-Free Algorithms

**Compare-and-Swap (CAS) Based Allocation**
```python
import threading

class LockFreeAllocator:
    def __init__(self, total_resources):
        self.resources = [AtomicBoolean(False) for _ in range(total_resources)]
        
    def allocate(self):
        max_attempts = len(self.resources) * 2
        
        for _ in range(max_attempts):
            # Random starting point
            start = random.randint(0, len(self.resources) - 1)
            
            for i in range(len(self.resources)):
                idx = (start + i) % len(self.resources)
                
                # Try to atomically claim resource
                if self.resources[idx].compare_and_set(False, True):
                    return idx
        
        return -1  # Allocation failed

class AtomicBoolean:
    def __init__(self, initial_value):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def compare_and_set(self, expected, new_value):
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False
```

### 3. Batch Processing

**Amortize Allocation Cost**
```python
class BatchAllocator:
    def __init__(self, base_allocator, batch_size=100):
        self.base = base_allocator
        self.batch_size = batch_size
        self.pending_requests = []
        self.lock = threading.Lock()
        
    def request_allocation(self, user_id, count, callback):
        with self.lock:
            self.pending_requests.append((user_id, count, callback))
            
            if len(self.pending_requests) >= self.batch_size:
                self.process_batch()
    
    def process_batch(self):
        # Sort requests for better locality
        requests = sorted(self.pending_requests, key=lambda x: x[0])
        self.pending_requests = []
        
        # Batch allocate
        results = self.base.batch_allocate(requests)
        
        # Invoke callbacks
        for (user_id, count, callback), result in zip(requests, results):
            callback(result)
```

## Real-World Examples

### 1. Kubernetes Pod Scheduling

Kubernetes uses sophisticated resource allocation for container placement:

```python
class KubernetesScheduler:
    def __init__(self):
        self.nodes = []  # Cluster nodes
        self.predicates = []  # Filters
        self.priorities = []  # Scoring functions
        
    def schedule_pod(self, pod):
        # Filter phase - find feasible nodes
        feasible_nodes = []
        for node in self.nodes:
            if all(pred(node, pod) for pred in self.predicates):
                feasible_nodes.append(node)
        
        if not feasible_nodes:
            return None  # No suitable node
        
        # Scoring phase - rank nodes
        scores = {}
        for node in feasible_nodes:
            score = sum(priority(node, pod) for priority in self.priorities)
            scores[node] = score
        
        # Select best node
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def add_predicate(self, pred_func):
        """Add filter like PodFitsResources, NodeSelector"""
        self.predicates.append(pred_func)
    
    def add_priority(self, priority_func):
        """Add scoring like LeastRequestedPriority, BalancedAllocation"""
        self.priorities.append(priority_func)
```

### 2. Database Connection Pooling

Connection pools manage expensive database connections:

```python
import queue
import threading
import time

class ConnectionPool:
    def __init__(self, min_size=5, max_size=20, host="localhost"):
        self.min_size = min_size
        self.max_size = max_size
        self.host = host
        
        self.pool = queue.Queue(maxsize=max_size)
        self.size = 0
        self.lock = threading.Lock()
        
        # Pre-create minimum connections
        for _ in range(min_size):
            self.pool.put(self._create_connection())
            self.size += 1
    
    def _create_connection(self):
        """Create new database connection"""
        return DatabaseConnection(self.host)
    
    def acquire(self, timeout=30):
        """Get connection from pool"""
        try:
            # Try to get existing connection
            conn = self.pool.get(block=False)
            
            # Validate connection
            if not conn.is_alive():
                conn = self._create_connection()
            
            return conn
            
        except queue.Empty:
            with self.lock:
                if self.size < self.max_size:
                    # Create new connection
                    self.size += 1
                    return self._create_connection()
            
            # Wait for available connection
            try:
                return self.pool.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("No connections available")
    
    def release(self, conn):
        """Return connection to pool"""
        if conn.is_alive():
            try:
                self.pool.put(conn, block=False)
            except queue.Full:
                # Pool is full, close connection
                conn.close()
                with self.lock:
                    self.size -= 1
        else:
            # Dead connection, don't return to pool
            with self.lock:
                self.size -= 1
```

### 3. Operating System Memory Management

Modern OS memory allocators like tcmalloc use sophisticated techniques:

```python
class TCMallocSimplified:
    def __init__(self):
        # Thread-local caches
        self.thread_caches = {}
        
        # Central free lists by size class
        self.central_lists = {
            8: [], 16: [], 32: [], 64: [], 128: [], 256: [],
            512: [], 1024: [], 2048: [], 4096: []
        }
        
        # Large object allocator
        self.page_heap = PageHeap()
    
    def malloc(self, size, thread_id):
        # Small allocation - use thread cache
        if size <= 256 * 1024:
            size_class = self.round_up_to_size_class(size)
            
            # Get thread cache
            if thread_id not in self.thread_caches:
                self.thread_caches[thread_id] = ThreadCache()
            
            cache = self.thread_caches[thread_id]
            
            # Try thread-local allocation
            ptr = cache.allocate(size_class)
            if ptr:
                return ptr
            
            # Refill from central list
            batch = self.central_lists[size_class][:32]
            if batch:
                self.central_lists[size_class] = self.central_lists[size_class][32:]
                cache.add_batch(size_class, batch[1:])
                return batch[0]
            
            # Allocate new span from page heap
            span = self.page_heap.allocate_span(size_class)
            objects = self.split_span_into_objects(span, size_class)
            cache.add_batch(size_class, objects[1:])
            return objects[0]
        
        # Large allocation - use page heap directly
        return self.page_heap.allocate_large(size)
    
    def free(self, ptr, size, thread_id):
        if size <= 256 * 1024:
            # Return to thread cache
            size_class = self.round_up_to_size_class(size)
            self.thread_caches[thread_id].deallocate(size_class, ptr)
        else:
            # Return to page heap
            self.page_heap.deallocate_large(ptr)
```

### 4. Cloud Resource Orchestration

AWS EC2 Spot Instance allocation uses market-based mechanisms:

```python
class SpotInstanceAllocator:
    def __init__(self):
        self.capacity_pools = {}  # (instance_type, az) -> available_capacity
        self.spot_prices = {}  # (instance_type, az) -> current_price
        self.bids = []  # Pending bids
        
    def request_spot_instances(self, request):
        """
        request = {
            'instance_types': ['t2.micro', 't2.small'],
            'max_price': 0.05,
            'count': 10,
            'allocation_strategy': 'lowest-price'
        }
        """
        
        if request['allocation_strategy'] == 'lowest-price':
            return self.allocate_lowest_price(request)
        elif request['allocation_strategy'] == 'capacity-optimized':
            return self.allocate_capacity_optimized(request)
        elif request['allocation_strategy'] == 'diversified':
            return self.allocate_diversified(request)
    
    def allocate_lowest_price(self, request):
        # Find cheapest capacity pools
        eligible_pools = []
        for (instance_type, az), price in self.spot_prices.items():
            if instance_type in request['instance_types'] and price <= request['max_price']:
                capacity = self.capacity_pools.get((instance_type, az), 0)
                if capacity > 0:
                    eligible_pools.append((price, instance_type, az, capacity))
        
        # Sort by price
        eligible_pools.sort()
        
        # Allocate from cheapest pools
        allocated = []
        remaining = request['count']
        
        for price, instance_type, az, capacity in eligible_pools:
            to_allocate = min(remaining, capacity)
            allocated.extend([(instance_type, az)] * to_allocate)
            remaining -= to_allocate
            
            if remaining == 0:
                break
        
        return allocated
    
    def allocate_capacity_optimized(self, request):
        # Allocate from pools with most capacity (least likely to be interrupted)
        eligible_pools = []
        for (instance_type, az), capacity in self.capacity_pools.items():
            if instance_type in request['instance_types']:
                price = self.spot_prices.get((instance_type, az), float('inf'))
                if price <= request['max_price']:
                    eligible_pools.append((capacity, instance_type, az))
        
        # Sort by capacity (descending)
        eligible_pools.sort(reverse=True)
        
        # Allocate from highest capacity pools
        allocated = []
        remaining = request['count']
        
        for capacity, instance_type, az in eligible_pools:
            to_allocate = min(remaining, capacity)
            allocated.extend([(instance_type, az)] * to_allocate)
            remaining -= to_allocate
            
            if remaining == 0:
                break
        
        return allocated
```

## Summary

Resource allocation is a fundamental problem with many sophisticated solutions. Key takeaways:

1. **Choose the Right Data Structure**: Bitmaps for dense allocation, lists for sparse, trees for hierarchical
2. **Algorithm Selection Matters**: First-fit for speed, best-fit for space efficiency, round-robin for fairness
3. **Consider Scale**: Different approaches work better at different scales
4. **Plan for Failure**: Include deadlock prevention, recovery mechanisms
5. **Optimize for Your Workload**: Batch processing, caching, and lock-free algorithms can dramatically improve performance
6. **Real Systems are Hybrid**: Combine multiple strategies based on runtime conditions

The optimal solution depends on your specific requirements:
- **Low latency**: Use deterministic allocation with minimal state
- **High throughput**: Implement lock-free algorithms with batching
- **Fairness**: Apply weighted fair queuing or round-robin
- **Space efficiency**: Choose best-fit with compaction
- **Reliability**: Implement Banker's algorithm for deadlock avoidance

Modern systems often combine multiple approaches, adapting their strategy based on current load, contention levels, and resource availability.