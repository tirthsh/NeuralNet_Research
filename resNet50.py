def unchanged_identity(ker_size, inputTensor, all_filters):
    #get the filters
    filter1, filter2, filter3 = all_filters
    
    #save input value so it can be used later
    identity = inputTensor
    
    #construct main path
    # First component
    inputTensor = Conv2D(filters = filter1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)
    inputTensor = Activation('relu')(inputTensor)
    
    # Second component
    inputTensor = Conv2D(filters = filter2, kernel_size = (ker_size, ker_size), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)
    inputTensor = Activation('relu')(inputTensor)

    # Third component
    inputTensor = Conv2D(filters = filter3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    inputTensor = Add()([inputTensor, identity])
    inputTensor = Activation('relu')(inputTensor)
   
    return inputTensor

def changed_identity(ker_size, inputTensor, all_filters, stride_int = 2):
    #get the filters
    filter1, filter2, filter3 = all_filters
    
    #save input value so it can be used later
    identity = inputTensor
    
    #construct main path
    # First component
    inputTensor = Conv2D(filters = filter1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)
    inputTensor = Activation('relu')(inputTensor)
    
    # Second component
    inputTensor = Conv2D(filters = filter2, kernel_size = (ker_size, ker_size), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)
    inputTensor = Activation('relu')(inputTensor)

    # Third component
    inputTensor = Conv2D(filters = filter3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(inputTensor)
    inputTensor = BatchNormalization(axis = 3)(inputTensor)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    identity = Conv2D(filters = filter3, kernal_size = (1,1), strides = (stride_int, stride_int), paddding = 'valid', kernel_initializer = glorot_uniform(seed=0))(identity)
    identity = BatchNormalization(axis = 3)(identity)
    
    inputTensor = Add()([inputTensor, identity])
    inputTensor = Activation('relu')(inputTensor)
   
    return inputTensor
