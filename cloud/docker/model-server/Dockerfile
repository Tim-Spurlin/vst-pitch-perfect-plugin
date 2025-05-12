FROM tensorflow/serving:latest-gpu

# Set environment variables
ENV MODEL_NAME=vocal_transformation_model
ENV MODEL_BASE_PATH=/models/vocal_transformation_model

# Copy the SavedModel
COPY ./models/${MODEL_NAME} /models/${MODEL_NAME}/1

# Set TensorFlow Serving configuration
COPY ./tensorflow_serving_config.conf /etc/tensorflow/tensorflow_serving_config.conf

# Set optimization settings for real-time inference
ENV TF_ENABLE_GPU_MEMORY_GROWTH=true
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=4
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
ENV TENSORFLOW_SESSION_PARALLELISM=4
ENV TENSORFLOW_INTRA_OP_PARALLELISM=4
ENV TENSORFLOW_INTER_OP_PARALLELISM=4

# Expose ports
EXPOSE 8500
EXPOSE 8501

# Run TensorFlow Serving with optimized settings
ENTRYPOINT ["tensorflow_model_server", \
            "--port=8500", \
            "--rest_api_port=8501", \
            "--model_name=${MODEL_NAME}", \
            "--model_base_path=${MODEL_BASE_PATH}", \
            "--enable_batching=true", \
            "--batching_parameters_file=/etc/tensorflow/batching_parameters.txt", \
            "--tensorflow_session_parallelism=${TENSORFLOW_SESSION_PARALLELISM}", \
            "--tensorflow_intra_op_parallelism=${TENSORFLOW_INTRA_OP_PARALLELISM}", \
            "--tensorflow_inter_op_parallelism=${TENSORFLOW_INTER_OP_PARALLELISM}"]