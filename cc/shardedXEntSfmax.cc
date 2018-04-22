#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;

REGISTER_OP("ShardedXentSfmax")
.Input("inputs: float32")
.Input("weights: float32")
.Input("biases: float32")
.Input("lower_bound: int32")
.Input("upper_bound: int32")
.Input("labels: int32")
.Output("batch_loss: float32");


class ShardedXentSfmaxOp : public OpKernel {
public:
  explicit ShardedXentSfmaxOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inputs = ctx->input(0);
    const Tensor& weights = ctx->input(1);
    const Tensor& biases = ctx->input(2);
    const Tensor& lower_bound = ctx->input(3);
    const Tensor& upper_bound = ctx->input(4);
    const Tensor& labels = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inputs.shape()),
		errors::InvalidArgument("shardedXEntSfmax : inputs is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weights.shape()),
		errors::InvalidArgument("shardedXEntSfmax: weights is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(biases.shape()),
		errors::InvalidArgument("shardedXEntSfmax: biases is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(lower_bound.shape()),
		errors::InvalidArgument("shardedXEntSfmax: lower_bound is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(upper_bound.shape()),
		errors::InvalidArgument("shardedXEntSfmax: upper_bound is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels.shape()),
		errors::InvalidArgument("shardedXEntSfmax: labels is not a vector"));
    OP_REQUIRES(ctx, (labels.dim_size(0) == lower_bound.dim_size(0)) &
		(lower_bound.dim_size(0) == upper_bound.dim_size(0)) &
		(upper_bound.dim_size(0) == inputs.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmax: inputs (i.e. batch), labels, lower & upper Bounds must have same length")
		);
    OP_REQUIRES(ctx, (weights.dim_size(1) == biases.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmax: weights & biases have incompatible dimensions"));
    OP_REQUIRES(ctx, (inputs.dim_size(1) == weights.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmax: inputs & weights have incompatible dimensions"));

    TensorShape batch_loss_shape;
    batch_loss_shape.AddDim(inputs.dim_size(0));
    Tensor* batch_loss = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, batch_loss_shape,
					  &batch_loss));
    
    auto batch_loss_vec = batch_loss->vec<float>();
    const auto& lvec = lower_bound.vec<int>();
    const auto& uvec = upper_bound.vec<int>();
    const auto& wmat = weights.matrix<float>();
    const auto& biasvec = biases.vec<float>();
    const auto& labvec = labels.vec<int>();
    const auto& invec = inputs.matrix<float>();
    for(int ib = 0; ib < inputs.dim_size(0); ib++) {
      // size of slice
      int slice_size = uvec(ib)-lvec(ib)+1;
      // for the product of inputs & wmat
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
      // to slice the inputs to row ib
      Eigen::array<int,2> offsets_invec = {ib,0};
      Eigen::array<int,2> extents_invec = {1,invec.dimensions()[1]};
      // to slice wmat on the set of valid labels
      Eigen::array<int,2> offsets_wmat = {0,lvec(ib)};
      Eigen::array<int,2> extents_wmat = {wmat.dimensions()[0],slice_size};
      // to slice the biasvec
      Eigen::array<int,1> offsets_bias = {lvec(ib)};
      Eigen::array<int,1> extents_bias = {slice_size};
      // to write the scratch result
      //auto  scratch = new Eigen::Tensor<float, 1,Eigen::RowMajor>(slice_size);
      auto  scratch = new Eigen::Tensor<float, 1,Eigen::RowMajor>();
      // evaluate the logits
      auto scratch_op0 = (invec.slice(offsets_invec,extents_invec))
	      .contract(wmat.slice(offsets_wmat, extents_wmat),product_dims)
	      .reshape(extents_bias) + biasvec.slice(offsets_bias,extents_bias);
      // subtract maximum from logits
      // first compute maximum, before broadcasting need to reshape scalar to a vec
      auto scratch_op1 = (scratch_op0.maximum().reshape(Eigen::array<int,1>{1}))
      .broadcast(extents_bias);
      auto scratch_op2 = scratch_op0 - scratch_op1;
      // take the exponential
      auto scratch_op3 = scratch_op2.exp();
      // take sum of exponential
      auto scratch_op4 = scratch_op3.sum().reshape(Eigen::array<int,1>{1}).broadcast(extents_bias);
      auto scratch_op5 = scratch_op3/scratch_op4;
      // this prompts the actual evaluation
      *scratch = scratch_op5;

      batch_loss_vec(ib) = scratch->operator()(labvec(ib)-lvec(ib));
      
      delete scratch;
   
  }
      batch_loss_vec = -batch_loss_vec.log();
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedXentSfmax").Device(DEVICE_CPU), ShardedXentSfmaxOp);
      
	  

      
      
    
    

