import tensorflow as tf

def check_gpu_availability():
    """
    Check if a GPU is available and return information about it.
    Returns:
        bool: True if GPU is available, False otherwise
        list: List of available GPUs
    """
    gpus = tf.config.list_physical_devices('GPU')
    is_gpu_available = len(gpus) > 0
    return is_gpu_available, gpus

def configure_gpu_memory_growth():
    """
    Configure GPU memory growth to avoid allocating all memory at once.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def set_gpu_memory_limit(limit_mb=None):
    """
    Set memory limit for GPU to avoid OOM errors.
    Args:
        limit_mb (int): Memory limit in megabytes. If None, no limit is set.
    """
    if limit_mb:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                )
            except RuntimeError as e:
                print(e)

def enable_mixed_precision():
    """
    Enable mixed precision training for faster training on GPUs.
    Mixed precision uses both float32 and float16 data types where appropriate.
    """
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def get_gpu_config():
    """
    Get optimal configurations for GPU training.
    Returns:
        dict: Dictionary containing GPU-optimized configurations
    """
    is_gpu_available, _ = check_gpu_availability()
    
    # Default configurations
    config = {
        'batch_size': 32,
        'batch_size_fine_tune': 16,
        'use_mixed_precision': False,
        'memory_limit': None  # No limit by default
    }
    
    if is_gpu_available:
        # GPU-optimized configurations
        config.update({
            'batch_size': 64,  # Larger batch size for GPU
            'batch_size_fine_tune': 32,
            'use_mixed_precision': True,
            'memory_limit': 4096  # 4GB limit, adjust based on your GPU
        })
    
    return config

def setup_gpu_training():
    """
    Set up all GPU configurations for optimal training.
    Returns:
        dict: GPU configuration settings
    """
    # Check GPU availability
    is_gpu_available, gpus = check_gpu_availability()
    if not is_gpu_available:
        print("No GPU detected. Training will proceed on CPU.")
        return get_gpu_config()

    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"- {gpu}")

    # Configure GPU memory growth
    configure_gpu_memory_growth()
    
    # Get optimal configurations
    config = get_gpu_config()
    
    # Set memory limit if specified
    if config['memory_limit']:
        set_gpu_memory_limit(config['memory_limit'])
    
    # Enable mixed precision if specified
    if config['use_mixed_precision']:
        enable_mixed_precision()
        print("Mixed precision training enabled")
    
    return config