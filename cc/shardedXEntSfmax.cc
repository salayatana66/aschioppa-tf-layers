#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <utility>
#include <map>

using namespace tensorflow;

REGISTER_OP("ShardedXentSfmaxNoGrad")
.Input("inputs: float32")
.Input("weights: float32")
.Input("biases: float32")
.Input("lower_bound: int32")
.Input("upper_bound: int32")
.Input("labels: int32")
.Output("batch_loss: float32");

class ShardedXentSfmaxNoGradOp : public OpKernel {
public:
  
  explicit ShardedXentSfmaxNoGradOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inputs = ctx->input(0);
    const Tensor& weights = ctx->input(1);
    const Tensor& biases = ctx->input(2);
    const Tensor& lower_bound = ctx->input(3);
    const Tensor& upper_bound = ctx->input(4);
    const Tensor& labels = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(inputs.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad : inputs is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weights.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: weights is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(biases.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: biases is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(lower_bound.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: lower_bound is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(upper_bound.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: upper_bound is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels.shape()),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: labels is not a vector"));
    OP_REQUIRES(ctx, (labels.dim_size(0) == lower_bound.dim_size(0)) &
		(lower_bound.dim_size(0) == upper_bound.dim_size(0)) &
		(upper_bound.dim_size(0) == inputs.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: inputs (i.e. batch), labels, lower & upper Bounds must have same length")
		);
    OP_REQUIRES(ctx, (weights.dim_size(0) == biases.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: weights & biases have incompatible dimensions"));
    OP_REQUIRES(ctx, (inputs.dim_size(1) == weights.dim_size(1)),
		errors::InvalidArgument("shardedXEntSfmaxNoGrad: inputs & weights have incompatible dimensions"));

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
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 1) };
      // to slice the inputs to row ib
      Eigen::array<int,2> offsets_invec = {ib,0};
      Eigen::array<int,2> extents_invec = {1,invec.dimensions()[1]};
      // to slice wmat on the set of valid labels
      Eigen::array<int,2> offsets_wmat = {lvec(ib),0};
      Eigen::array<int,2> extents_wmat = {slice_size,wmat.dimensions()[1]};
      // to slice the biasvec
      Eigen::array<int,1> offsets_bias = {lvec(ib)};
      Eigen::array<int,1> extents_bias = {slice_size};
      // to write the scratch result
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

REGISTER_KERNEL_BUILDER(Name("ShardedXentSfmaxNoGrad").Device(DEVICE_CPU), ShardedXentSfmaxNoGradOp);

REGISTER_OP("ShardedXentSfmax")
.Input("inputs: float32")
.Input("weights: float32")
.Input("biases: float32")
.Input("lower_bound: int32")
.Input("upper_bound: int32")
.Input("labels: int32")
.Output("batch_loss: float32")
.Output("grad_inputs: float32")
.Output("grad_weights_indices: int32")
.Output("grad_weights_values: float32")
.Output("grad_biases_indices: int32")
.Output("grad_biases_values: float32");

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
    OP_REQUIRES(ctx, (weights.dim_size(0) == biases.dim_size(0)),
		errors::InvalidArgument("shardedXEntSfmax: weights & biases have incompatible dimensions"));
    OP_REQUIRES(ctx, (inputs.dim_size(1) == weights.dim_size(1)),
		errors::InvalidArgument("shardedXEntSfmax: inputs & weights have incompatible dimensions"));

    TensorShape batch_loss_shape;
    batch_loss_shape.AddDim(inputs.dim_size(0));
    Tensor* batch_loss = NULL;
    // allocate the output for the batch_loss
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, batch_loss_shape,
					  &batch_loss));
    Tensor* grad_inputs = NULL;
    // allocate the output for the gradient wrt to the inputs
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({inputs.dim_size(0),inputs.dim_size(1)}),
					  &grad_inputs));

    // get references to inputs

    const auto& lvec = lower_bound.vec<int>();
    const auto& uvec = upper_bound.vec<int>();
    const auto& wmat = weights.matrix<float>();
    const auto& biasvec = biases.vec<float>();
    const auto& labvec = labels.vec<int>();
    const auto& invec = inputs.matrix<float>();
    // compute how big the sparse gradients will be
    int weights_grad_size = 0;
    int biases_grad_size = 0;
    for(int ib = 0; ib < inputs.dim_size(0); ib++) {
      int slice_size = uvec(ib)-lvec(ib)+1;
      biases_grad_size += slice_size;
      weights_grad_size += (int)inputs.dim_size(1)*slice_size;
    }
    
    Tensor* grad_weights_indices = NULL;
    // allocate the output for the gradient wrt to the weights (this update is sparse!!)
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({weights_grad_size,3}),
					  &grad_weights_indices));
    Tensor* grad_weights_values = NULL;
    // allocate the output for the gradient wrt to the weights (this update is sparse!!)
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({weights_grad_size}),
					  &grad_weights_values));

    Tensor* grad_biases_indices = NULL;
    // allocate the output for the gradient wrt to the biases (this update is sparse !!)
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, TensorShape({biases_grad_size,2}),
					     &grad_biases_indices));
    Tensor* grad_biases_values = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(5, TensorShape({biases_grad_size}),
					     &grad_biases_values));

    // get reference to outputs
    auto batch_loss_vec = batch_loss->vec<float>();
    auto grad_inputs_mat = grad_inputs->matrix<float>();
    auto grad_biases_indices_mat = grad_biases_indices->matrix<int>();
    auto grad_biases_values_vec = grad_biases_values->vec<float>();
    auto grad_weights_indices_mat = grad_weights_indices->matrix<int>();
    auto grad_weights_values_vec = grad_weights_values->vec<float>();
    // instantiate counters for the indices of gradients wrt to weights & biases
    int iwg =0;
    int bwg = 0;
    for(int ib = 0; ib < inputs.dim_size(0); ib++) {
      // size of slice
      int slice_size = uvec(ib)-lvec(ib)+1;
      // for the product of inputs & wmat
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 1) };
      // to slice the inputs to row ib
      Eigen::array<int,2> offsets_invec = {ib,0};
      Eigen::array<int,2> extents_invec = {1,invec.dimensions()[1]};
      // to slice wmat on the set of valid labels
      Eigen::array<int,2> offsets_wmat = {lvec(ib),0};
      Eigen::array<int,2> extents_wmat = {slice_size,wmat.dimensions()[1]};
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
      // scratch holds the probabilities for the different classes
      *scratch = scratch_op5;

      // remember that we still need to take the logarithm for the loss
      batch_loss_vec(ib) = scratch->operator()(labvec(ib)-lvec(ib));

      // allocate for the gradient of the probability wrt. to the scores
      auto pgrad = new Eigen::Tensor<float,1,Eigen::RowMajor>(slice_size);
      for(int c0 = 0; c0 < slice_size; c0++) 
	pgrad->operator()(c0) = -scratch->operator()(c0) * scratch->operator()(labvec(ib)-lvec(ib));
      pgrad->operator()(labvec(ib)-lvec(ib)) += scratch->operator()(labvec(ib)-lvec(ib));
      auto igrad = new Eigen::Tensor<float,1,Eigen::RowMajor>(inputs.dim_size(1));
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims_igrad = { Eigen::IndexPair<int>(0, 0) };
      *igrad = wmat.slice(offsets_wmat, extents_wmat).contract(*pgrad,product_dims_igrad);

      for(int a0 = 0; a0 < inputs.dim_size(1); a0++) {
	grad_inputs_mat(ib,a0) = -1.0/scratch->operator()(labvec(ib)-lvec(ib))*igrad->operator()(a0);
      }
      for(int c0 = 0; c0 < slice_size; c0++) {
	grad_biases_indices_mat(bwg,0) = ib;
	grad_biases_indices_mat(bwg,1) = c0+lvec(ib);
	grad_biases_values_vec(bwg) = -1.0/scratch->operator()(labvec(ib)-lvec(ib)) * (pgrad->operator()(c0));
	bwg++;
      }

      for(int a0 = 0; a0 < inputs.dim_size(1); a0++) {
	for(int c0 = 0; c0 < slice_size; c0++) {
	  grad_weights_indices_mat(iwg,0) = ib;
	  grad_weights_indices_mat(iwg,1) = a0;
	  grad_weights_indices_mat(iwg,2) = c0+lvec(ib);
          grad_weights_values_vec(iwg) = -1.0/scratch->operator()(labvec(ib)-lvec(ib))
	   * (pgrad->operator()(c0)) *
	  invec(ib,a0);
	  iwg++;
	}
      }
      
      // free resources
      delete scratch;
      delete pgrad;
      delete igrad;
   
  }
    // remember to take the logarithm to get the loss
      batch_loss_vec = -batch_loss_vec.log();
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedXentSfmax").Device(DEVICE_CPU), ShardedXentSfmaxOp);

REGISTER_OP("ShardedXentSfmaxHelperGrad")
.Input("in_grad_loss: float32")
.Input("in_grad_inputs: float32")
.Input("in_grad_weights_indices: int32")
.Input("in_grad_weights_values: float32")
.Input("in_grad_biases_indices: int32")
.Input("in_grad_biases_values: float32")
.Output("out_grad_inputs: float32")
.Output("out_grad_weights_indices: int32")
.Output("out_grad_weights_values: float32")
.Output("out_grad_biases_indices: int32")
.Output("out_grad_biases_values: float32");

class ShardedXentSfmaxHelperGradOp : public OpKernel {
public:
  explicit ShardedXentSfmaxHelperGradOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inGradLoss = ctx->input(0);
    const Tensor& inGradInputs= ctx->input(1);
    const Tensor& inGradWI= ctx->input(2);
    const Tensor& inGradWV = ctx->input(3);
    const Tensor& inGradBI = ctx->input(4);
    const Tensor& inGradBV = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(inGradLoss.shape()),
		errors::InvalidArgument("Gradient of shardedXEntSfmax : \
                                         incoming loss gradient is not a vector"));
    

    Tensor* out_grad_inputs = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({inGradInputs.dim_size(0),
	      inGradInputs.dim_size(1)})
					     , &out_grad_inputs));

    // get references to inputs
    const auto& inGradLossVec = inGradLoss.vec<float>();
    const auto& inGradInputsMat = inGradInputs.matrix<float>();
    const auto& inGradWIMat = inGradWI.matrix<int>();
    const auto& inGradWVVec = inGradWV.vec<float>();
    const auto& inGradBIMat = inGradBI.matrix<int>();
    const auto& inGradBVVec = inGradBV.vec<float>();
    // the output gradient wrt to the inputs
    auto out_grad_inputs_mat = out_grad_inputs->matrix<float>();
    /*
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(0, 0) };
    out_grad_inputs_vec = inGradLossVec.contract(inGradInputsMat, product_dims);
    */
    Eigen::array<int, 2> bcast({1, inGradInputs.dim_size(1)});
    out_grad_inputs_mat = inGradLossVec.reshape(Eigen::array<int,2>{inGradLoss.dim_size(0),
    	  1}).broadcast(bcast) * inGradInputsMat;
    // the output gradient with respect to the weights
    // by default map constructor does the right thing with
    // pairs of integers unseen: the float is automatically initialized at 0
    // we also use a second map to store the distinct sparse indices that we see
    std::map<std::pair<int,int>,float> WMap;
    std::map<int,int> distinctWMap;
    // product of sparse & dense using a map
    for(int i = 0; i < inGradWI.dim_size(0); ++i) {
      WMap[std::make_pair(inGradWIMat(i,1), inGradWIMat(i,2))] +=
	inGradLossVec(inGradWIMat(i,0))*inGradWVVec(i);
      distinctWMap[inGradWIMat(i,2)] += 1;
    }
    
    
    Tensor* out_grad_weights_indices = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({distinctWMap.size()})
					     , &out_grad_weights_indices));
    Tensor* out_grad_weights_values = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({distinctWMap.size(),
	      inGradInputs.dim_size(1)})
					     , &out_grad_weights_values));
    auto out_grad_weight_indices_vec = out_grad_weights_indices->vec<int>();
    auto out_grad_weights_values_mat = out_grad_weights_values->matrix<float>();
    // transverse the maps
    int wib = 0;
    for(auto it = distinctWMap.cbegin(); it != distinctWMap.cend(); ++it) {
      out_grad_weight_indices_vec(wib) = it->first;
      for(int a0 = 0; a0 < inGradInputs.dim_size(1); a0++) {
	out_grad_weights_values_mat(wib,a0) = WMap[std::make_pair(a0,it->first)];
      }
      wib++;
    }

    // the output gradient with respect to the weights
    // by default map constructor does the right thing with
    // pairs of integers unseen: the float is automatically initialized at 0
    std::map<int,float> BMap;
    // product of sparse & dense using a map
    for(int i = 0; i < inGradBI.dim_size(0); ++i) {
      BMap[inGradBIMat(i,1)] += inGradLossVec(inGradBIMat(i,0))*inGradBVVec(i);
    }

    Tensor* out_grad_bias_indices = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({BMap.size()})
					     , &out_grad_bias_indices));

    Tensor* out_grad_bias_values = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, TensorShape({BMap.size()})
					     , &out_grad_bias_values));
    auto out_grad_bias_indices_vec = out_grad_bias_indices->vec<int>();
    auto out_grad_bias_values_vec = out_grad_bias_values->vec<float>();
    
    // to transverse along with the map
    int bib = 0;
    for(auto it = BMap.cbegin(); it != BMap.cend(); ++it) {
      out_grad_bias_indices_vec(bib) = it->first;
      out_grad_bias_values_vec(bib) = it->second;
      bib++;
      }


  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedXentSfmaxHelperGrad").Device(DEVICE_CPU),
			ShardedXentSfmaxHelperGradOp);

		
