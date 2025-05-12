#!/usr/bin/env python3
"""
Export trained vocal transformation models for deployment
Optimized for real-time inference in production environments
"""

import os
import argparse
import tensorflow as tf
from model import create_voice_transformation_model, export_model_for_inference, convert_to_tflite

def optimize_for_inference(model):
    """Apply TensorFlow optimizations for inference speed"""
    # Convert variables to constants for faster inference
    frozen_func = tf.function(lambda x: model(x))
    frozen_func = frozen_func.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    
    # Apply grappler optimizations
    optimizer = tf.compat.v1.OptimizerOptions(
        opt_level=tf.compat.v1.OptimizerOptions.L1,
        do_common_subexpression_elimination=True,
        do_constant_folding=True,
        do_function_inlining=True)
    
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.CopyFrom(optimizer)
    
    return frozen_func

def main():
    parser = argparse.ArgumentParser(description="Export vocal transformation models for deployment")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.h5 file)")
    parser.add_argument("--output_dir", type=str, default="./exported_models",
                        help="Directory to save exported models")
    parser.add_argument("--model_name", type=str, default="vocal_transformation",
                        help="Name for the exported model")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization for smaller model size")
    parser.add_argument("--optimize", action="store_true",
                        help="Apply additional optimizations for inference")
    parser.add_argument("--target", choices=["serving", "tflite", "all"], default="all",
                        help="Target format for export")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Apply optimizations if requested
    if args.optimize:
        print("Applying inference optimizations...")
        model = optimize_for_inference(model)
    
    # Export for TensorFlow Serving
    if args.target in ["serving", "all"]:
        serving_dir = os.path.join(args.output_dir, f"{args.model_name}_serving")
        print(f"Exporting model for TensorFlow Serving to {serving_dir}...")
        export_model_for_inference(model, serving_dir)
    
    # Export to TFLite
    if args.target in ["tflite", "all"]:
        tflite_path = os.path.join(args.output_dir, f"{args.model_name}.tflite")
        print(f"Exporting model to TFLite format with quantization={args.quantize}...")
        convert_to_tflite(model, tflite_path, quantize=args.quantize)
    
    print("Export complete!")

if __name__ == "__main__":
    main()