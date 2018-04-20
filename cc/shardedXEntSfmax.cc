#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

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
      int i_start = lvec(ib);
      int i_width = uvec(ib)-lvec(ib);
      TensorShape* scratch_shape = new TensorShape;
      scratch_shape->AddDim(i_width);
      Tensor* scratch = new Tensor(DT_FLOAT, *scratch_shape);
      auto scratch_vec = scratch->vec<float>();
      // initialize everything to 0 values... is it necessary?
      for(int j = 0; j < i_width; j++) {
	scratch_vec(j) = 0;
	// dot product with the invec
	for(int l = 0; l < inputs.dim_size(1); ++l) {
	  scratch_vec(j) += invec(ib,l)*wmat(l,j+i_start);
	}
	// add the biases
	scratch_vec(j) += biasvec(j+i_start);
      }
      // for numerical stability we need to sutract the maximum from the logits
      auto maxCoeff = scratch_vec(0);
      for(int l = 0; l < inputs.dim_size(1); ++l) {
	if(maxCoeff > scratch_vec(l)) maxCoeff == scratch_vec(l);
      }
      for(int l = 0; l < inputs.dim_size(1); ++l)
	scratch_vec(l) -= maxCoeff; 

      Tensor* exp_scratch = new Tensor(DT_FLOAT, *scratch_shape);
      auto exp_scratch_vec = exp_scratch->vec<float>();
      exp_scratch_vec = scratch_vec.exp();
      auto denom = exp_scratch_vec(0);
      for(int l = 1; l<inputs.dim_size(1); ++l)
	denom += exp_scratch_vec(l);
      for(int l =0; l<inputs.dim_size(1); ++l)
	exp_scratch_vec(l) = exp_scratch_vec(l)/denom;
      batch_loss_vec(ib) = exp_scratch_vec(labvec(ib)-i_start);
      
      
      delete exp_scratch;
      delete scratch;
      delete scratch_shape; 
  }
    //batch_loss_vec = batch_loss_vec.log();
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedXentSfmax").Device(DEVICE_CPU), ShardedXentSfmaxOp);
      
	  

      
      
    
    

