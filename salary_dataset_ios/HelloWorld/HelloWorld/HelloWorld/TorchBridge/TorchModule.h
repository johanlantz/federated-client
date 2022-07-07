#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)predict:(const void*)buffer withXDim:(int) xDim andYDim:(int) yDim NS_SWIFT_NAME(predict(buffer:xDim:yDim));

- (void)train:(const void*)buffer withXDim:(int) xDim andYDim:(int) yDim NS_SWIFT_NAME(train(buffer:xDim:yDim));

@end

NS_ASSUME_NONNULL_END
