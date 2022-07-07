#import "TorchModule.h"
#import <LibTorch/LibTorch.h>

@implementation TorchModule {
 @protected
  torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      auto qengines = at::globalContext().supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
      }
      _impl = torch::jit::load(filePath.UTF8String);
        
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (NSArray<NSNumber*>*)predict:(const void*)buffer withXDim:(int)xDim andYDim:(int) yDim {
  try {
    at::Tensor tensor = torch::from_blob((void*)buffer, {xDim, yDim}, at::kFloat);
      NSLog(@"Hello %zu", tensor.numel());
    _impl.eval();
//      
//      auto foo_a = tensor.accessor<float,1>();
//      for(int i = 0; i < foo_a.size(0); i++) {
//        // use the accessor foo_a to get tensor data.
//        
//          NSLog(@"Hello %f", foo_a[i]);
//      }
      
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto outputTensor = _impl.forward({tensor}).toTensor();
      NSLog(@"output tensor %lld", outputTensor.numel());
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < outputTensor.numel(); i++) {
      [results addObject:@(floatBuffer[i])];
      NSLog(@"Adding result: %f", floatBuffer[i]);
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

//TBI if training is even supported on mobile
- (void)train:(const void*)buffer withXDim:(int)xDim andYDim:(int) yDim {
//  try {
//    at::Tensor tensor = torch::from_blob((void*)buffer, {xDim, yDim}, at::kFloat);
//      NSLog(@"Hello %zu", tensor.numel());
//      //_impl.dump(false, false, true);
//      torch::jit::script::parameter_list params = _impl.parameters();
//      NSLog(@"hello");
//      _impl.save("");
//      
//      return;
//      _impl.train();
//      
// 
////
////      auto foo_a = tensor.accessor<float,1>();
////      for(int i = 0; i < foo_a.size(0); i++) {
////        // use the accessor foo_a to get tensor data.
////
////          NSLog(@"Hello %f", foo_a[i]);
////      }
//      
//    torch::autograd::AutoGradMode guard(true);
//    //at::AutoNonVariableTypeMode non_var_type_mode(true);
//    auto outputTensor = _impl.forward({tensor}).toTensor();
//      NSLog(@"output tensor %lld", outputTensor.numel());
//    float* floatBuffer = outputTensor.data_ptr<float>();
//    if (!floatBuffer) {
//      return;
//    }
//    NSMutableArray* results = [[NSMutableArray alloc] init];
//    for (int i = 0; i < outputTensor.numel(); i++) {
//      [results addObject:@(floatBuffer[i])];
//      NSLog(@"Adding result: %f", floatBuffer[i]);
//    }
//    return;
//  } catch (const std::exception& exception) {
//    NSLog(@"%s", exception.what());
//  }
  return;
}

//- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
//  try {
//    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
//    torch::autograd::AutoGradMode guard(false);
//    at::AutoNonVariableTypeMode non_var_type_mode(true);
//    auto outputTensor = _impl.forward({tensor}).toTensor();
//    float* floatBuffer = outputTensor.data_ptr<float>();
//    if (!floatBuffer) {
//      return nil;
//    }
//    NSMutableArray* results = [[NSMutableArray alloc] init];
//    for (int i = 0; i < 1000; i++) {
//      [results addObject:@(floatBuffer[i])];
//    }
//    return [results copy];
//  } catch (const std::exception& exception) {
//    NSLog(@"%s", exception.what());
//  }
//  return nil;
//}

@end

