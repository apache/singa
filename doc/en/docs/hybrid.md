# Hybrid Parallelism

---

## User Guide

SINGA supports different parallelism options for distributed training.
Users just need to configure it in the job configuration.

Both `NetProto` and `LayerProto` have a field `partition_dim` to control the parallelism option:

  * `partition_dim=0`: neuralnet/layer is partitioned on data dimension, i.e., each worker processes a subset of data records.
  * `partition_dim=1`: neuralnet/layer is partitioned on feature dimension, i.e., each worker maintains a subset of feature parameters.

`partition_dim` field in `NetProto` will be applied to all layers, unless a layer has its own `partition_dim` field set.

If we want data parallelism for the whole model, just leave `partition_dim` as default (which is 0), or configure the job.conf like:

```
neuralnet {
  partition_dim: 0
  layer {
    name: ... 
    type: ...
  }
  ...
}
```

With the hybrid parallelism, we can have layers either partitioned on data dimension or feature dimension.
For example, if we want a specific layer partitioned on feature dimension, just configure like:

```
neuralnet {
  partition_dim: 0
  layer {
    name: "layer1_partition_on_data_dimension"
    type: ...
  }
  layer {
    name: "layer2_partition_on_feature_dimension"
    type: ...
    partition_dim: 1
  }
  ...
}
```

## Developer Guide

To support hybrid parallelism, after singa read users' model and paration configuration, a set of connection layers are automatically added between layers when needed:

* `BridgeSrcLayer` & `BridgeDstLayer` are added when two connected layers are not in the same machine. They are paired and are responsible for sending data/gradient to the other side during each iteration.

* `ConcateLayer` is added when there are multiple source layers. It combines their feature blobs along a given dimension.

* `SliceLayer` is added when there are mutliple dest layers, each of which only needs a subset(slice) of this layers' feature blob.

* `SplitLayer` is added when there are multiple dest layers, each of which needs the whole feature blob.

Following is the logic used in our code to add connection layers:

```
Add Slice, Concate, Split Layers for Hybrid Partition

All cases are as follows:
src_pdim | dst_pdim | connection_type | Action
    0    |     0    |     OneToOne    | Direct Connection
    1    |     1    |     OneToOne    | Direct Connection
    0    |     0    |     OneToAll    | Direct Connection
    1    |     0    |     OneToOne    | Slice -> Concate
    0    |     1    |     OneToOne    | Slice -> Concate
    1    |     0    |     OneToAll    | Slice -> Concate
    0    |     1    |     OneToAll    | Split -> Concate
    1    |     1    |     OneToAll    | Split -> Concate

Logic:
dst_pdim = 1 && OneToAll ?
  (YES) Split -> Concate
  (NO)  src_pdim = dst_pdim ?
          (YES) Direct Connection
          (NO)  Slice -> Concate
```
