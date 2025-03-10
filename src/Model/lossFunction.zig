const std = @import("std");
const zant = @import("../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const Convert = zant.utils.type_converter;
//import error library
const error_handler = zant.utils.error_handler;
const TensorError = error_handler.TensorError;
const TensorMathError = error_handler.TensorMathError;
const LossError = error_handler.LossError;

const pkg_allocator = zant.utils.allocator.allocator;

/// possible Types of Loss function.
/// Every time a new loss function is added you must update the enum
pub const LossType = enum {
    MSE,
    CCE,
};

/// LossFunction Interface.
/// Is used to initialize a generic loss function. Every time you want to add a Loss Function the switch must be updated.
pub fn LossFunction(lossType: LossType) type {
    const ls = switch (lossType) {
        LossType.MSE => MSELoss(),
        LossType.CCE => CCELoss(),
    };

    return struct {
        loss: type = ls,

        /// Return a rensor where the smallest element is the result of the loss function of each row of "predicitions" wrt "targets".
        ///  ex:
        /// PredictionTens =[ [ vect , vect ],
        ///                   [ vect , vect ],
        ///                   [ vect , vect ] ] -> 3 x 2 x vect.len
        /// TargetTens = same of prediction
        /// OutputTens = [ [ a, b],
        ///                [ c, d],
        ///                [ e, f] ] -> 3 x 2 where each letter is the result of the loss function applied to the "vect" of predictions and targets
        ///
        /// Abstract Method called to compute the loss of a prediction
        pub fn computeLoss(self: *const @This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            return try self.loss.computeLoss(T, predictions, targets);
        }

        /// Abstract Method called to compute the gradient of the loss of a prediction
        pub fn computeGradient(self: *const @This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            return try self.loss.computeGradient(T, predictions, targets);
        }
    };
}

/// MSE measures the average squared difference between the predicted values and the actual target values.
/// MSE penalizes larger errors more than smaller ones due to the squaring of differences, making it sensitive to outliers.
pub fn MSELoss() type {
    return struct {
        pub fn computeLoss(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            try basicChecks(T, predictions);
            try basicChecks(T, targets);

            //CHECKS :
            //   -size matching
            if (predictions.size != targets.size) return TensorMathError.InputTensorDifferentSize;

            //create the shape of the output tensor,
            //OSS: its len is predictions.shape.len - 1
            const allocator = predictions.allocator;
            var out_shape = allocator.alloc(usize, (predictions.shape.len - 1)) catch {
                return TensorMathError.MemError;
            };
            defer allocator.free(out_shape);

            //filling shape
            for (0..out_shape.len) |i| {
                out_shape[i] = predictions.shape[i];
            }

            //initializing the output Tensor containing the result of the forward
            var out_tensor = Tensor(T).fromShape(allocator, out_shape) catch {
                return TensorMathError.MemError;
            };

            //initialize the current location to all 0
            const location = allocator.alloc(usize, predictions.shape.len) catch {
                return TensorMathError.MemError;
            };
            defer allocator.free(location);
            for (location) |*loc| {
                loc.* = 0;
            }

            //call mutidim_mat_mul to handle multidimensionality
            try multidim_MSE(
                T,
                predictions,
                targets,
                &out_tensor,
                0,
                location,
            );

            //out_tensor.info();
            return out_tensor;
        }

        /// Method used to handle multidimensionality in tensors. It could be generalised into a unique global function that in last place calls
        /// a function, passed by argument, to compute the loss.
        fn multidim_MSE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void {
            if (current_depth == (predictions.shape.len - 1)) {
                //declaring res as the result of the sum of the MSE
                var res: T = 0;
                const allocator = predictions.allocator;

                //get_location is just used to handle multidimensionality in an easy way. You can see it as coordinates in a multidimensional place.
                const get_location = try allocator.alloc(usize, location.len);
                defer allocator.free(get_location);

                //initializing get location to the same values of location
                for (0..get_location.len) |i| {
                    get_location[i] = location[i];
                }
                //calculating the loss
                for (0..predictions.shape[current_depth]) |i| {
                    get_location[current_depth] = i; //for each element of predictions vect and target vect

                    const target = try targets.get_at(location);
                    const prediction = try predictions.get_at(location);
                    const diff = target - prediction;
                    res += diff * diff;
                }
                const divisor: T = Convert.convert(usize, T, predictions.shape[current_depth]);

                switch (@typeInfo(T)) {
                    .Int => res = @divFloor(res, divisor),
                    else => res = res / divisor,
                }

                //declaring and initializing the landing location of the sum
                const out_location = try allocator.alloc(usize, predictions.shape.len - 1);
                defer allocator.free(out_location);

                for (0..out_location.len) |i| {
                    out_location[i] = location[i];
                }

                //set the loss value into out_tensor
                try out_tensor.set_at(out_location, res);
            } else { //otherwise I have to go deepers
                for (0..predictions.shape[current_depth]) |element_at_current_depth| {
                    location[current_depth] = element_at_current_depth;

                    try multidim_MSE(
                        T,
                        predictions,
                        targets,
                        out_tensor,
                        current_depth + 1,
                        location,
                    );
                }
            }
        }

        pub fn computeGradient(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            try basicChecks(T, predictions);
            try basicChecks(T, targets);

            //check on the size of predictions, targets and gradient
            if (predictions.size != targets.size) {
                return LossError.SizeMismatch;
            }
            //checks on the shape of predictions, targets and gradient
            if (predictions.shape.len != targets.shape.len) {
                return LossError.ShapeMismatch;
            }
            for (predictions.shape, 0..) |*dim, i| {
                if (dim.* != targets.shape[i]) return LossError.ShapeMismatch;
            }

            var gradient = try Tensor(T).fromShape(predictions.allocator, predictions.shape);
            const n: f32 = @floatFromInt(predictions.size);

            for (0..gradient.size) |i| {
                gradient.data[i] = (2.0 / n) * (predictions.data[i] - targets.data[i]);
            }

            return gradient;
        }
    };
}

/// Categorical Cross-Entropy (CCE) is a loss function commonly used in classification tasks, particularly for multi-class problems.
/// It measures the dissimilarity between the true label distribution and the predicted probability distribution output by the model.
pub fn CCELoss() type {
    return struct {
        //return a rensor where the smallest element is the result of the loss function for each array of weights
        //         //ex:
        //         // PredictionTens =[ [ vect , vect ],
        //         //                   [ vect , vect ],
        //         //                   [ vect , vect ] ] -> 3 x 2 x vect.len
        //         // TargetTens = same of prediction
        //         // OutputTens = [ [ a, b],
        //         //                [ c, d],
        //         //                [ e, f] ] -> 3 x 2 where each letter is the result of the loss function applied to the "vect" of predictions and targets
        //         //
        //
        pub fn computeLoss(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            try basicChecks(T, predictions);
            try basicChecks(T, targets);

            //CHECKS :
            // - size matching
            if (predictions.size != targets.size) return TensorMathError.InputTensorDifferentSize;

            //create the shape of the output tensor,
            //OSS: its len is predictions.shape.len - 1
            const allocator = predictions.allocator;
            var out_shape = allocator.alloc(usize, (predictions.shape.len - 1)) catch {
                return TensorMathError.MemError;
            };
            defer allocator.free(out_shape);

            for (0..out_shape.len) |i| {
                out_shape[i] = predictions.shape[i];
            }

            var out_tensor = Tensor(T).fromShape(allocator, out_shape) catch {
                return TensorMathError.MemError;
            };

            //initialize the current location to all 0
            const location = allocator.alloc(usize, predictions.shape.len) catch {
                return TensorMathError.MemError;
            };
            defer allocator.free(location);
            for (location) |*loc| {
                loc.* = 0;
            }

            //call mutidim_mat_mul to handle multidimensionality
            try multidim_CCE(
                T,
                predictions,
                targets,
                &out_tensor,
                0,
                location,
            );

            return out_tensor;
        }

        fn multidim_CCE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void {
            if (current_depth == (predictions.shape.len - 1)) {
                var res: f64 = 0.0;
                const epsilon = 1e-15; // Smaller epsilon for more precision

                const get_location = try pkg_allocator.alloc(usize, location.len);
                defer pkg_allocator.free(get_location);

                for (0..get_location.len) |i| {
                    get_location[i] = location[i];
                }

                // Calculate loss with numerical stability
                for (0..predictions.shape[current_depth]) |i| {
                    get_location[current_depth] = i;
                    const target = try targets.get_at(get_location);
                    var prediction = try predictions.get_at(get_location);

                    // Clip predictions to avoid numerical instability
                    prediction = @max(epsilon, @min(1.0 - epsilon, prediction));

                    // Use log with clipped value
                    const log = std.math.log(f64, std.math.e, prediction);
                    res -= (target * log);
                }

                const out_location = try pkg_allocator.alloc(usize, predictions.shape.len - 1);
                defer pkg_allocator.free(out_location);

                for (0..out_location.len) |i| {
                    out_location[i] = location[i];
                }

                const out_res: T = Convert.convert(f64, T, res);
                try out_tensor.set_at(out_location, out_res);
            } else {
                for (0..predictions.shape[current_depth]) |element_at_current_depth| {
                    location[current_depth] = element_at_current_depth;
                    try multidim_CCE(T, predictions, targets, out_tensor, current_depth + 1, location);
                }
            }
        }

        pub fn computeGradient(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            try basicChecks(T, predictions);
            try basicChecks(T, targets);

            // Check size
            if (predictions.size != targets.size) {
                return LossError.SizeMismatch;
            }

            // Check shape
            if (predictions.shape.len != targets.shape.len) {
                return LossError.ShapeMismatch;
            }
            for (predictions.shape, 0..) |*dim, i| {
                if (dim.* != targets.shape[i]) return LossError.ShapeMismatch;
            }

            const n = predictions.size;
            var gradient = try Tensor(T).fromShape(predictions.allocator, predictions.shape);

            // Define a small epsilon to avoid division by zero
            const epsilon = @as(T, 1e-12);

            for (0..n) |i| {
                // Add epsilon to avoid division by zero
                gradient.data[i] = -targets.data[i] / (predictions.data[i] + epsilon);
            }

            return gradient;
        }
    };
}

fn basicChecks(comptime T: anytype, tensor: *Tensor(T)) !void {

    //not empty data
    if (tensor.data.len == 0 or std.math.isNan(tensor.data.len)) {
        return TensorError.EmptyTensor;
    }

    //not zero shape
    if (tensor.shape.len == 0 or std.math.isNan(tensor.data.len)) {
        return TensorError.EmptyTensor;
    }

    //real size
    if (tensor.size == 0 or std.math.isNan(tensor.size)) {
        return TensorError.ZeroSizeTensor;
    }
}
