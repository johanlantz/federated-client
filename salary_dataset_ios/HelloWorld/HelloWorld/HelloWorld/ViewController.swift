import UIKit

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "model_script", ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()
    
    private lazy var testData: [[Float32]] = {
        if let filePath = Bundle.main.path(forResource: "X_test_norm", ofType: "csv"),
            let fileData = try? String(contentsOfFile: filePath) {
            var xRows = fileData.components(separatedBy: .newlines)
            // Remove newline(s) at the end of the file
            xRows.removeAll(where: {row in
                return row.count > 1 ? false : true
            })
            
            let xRowsFloat = xRows.map { $0.components(separatedBy: ",").map {Float32($0)!}}
            return xRowsFloat
        } else {
            fatalError("Can't find the text file!")
        }
    }()
    
    private lazy var testLabels: [Int] = {
        if let filePath = Bundle.main.path(forResource: "y_test_norm", ofType: "csv"),
            let fileData = try? String(contentsOfFile: filePath) {
            var yRows = fileData.components(separatedBy: .newlines)
            // Remove newline(s) at the end of the file
            yRows.removeAll(where: {row in
                return row.count > 0 ? false : true
            })
            
            let yRowsInt = yRows.map {Int($0)!}
            return yRowsInt
        } else {
            fatalError("Can't find the text file!")
        }
    }()
    
    private func trainModel() {
        module.train(testData[0], withXDim: 1, andYDim: Int32(testData[0].count))
    }
    
    
    //private func validate
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
//        trainModel()
//        return
        
        //Important, we need to work with Float32 otherwise it does not match the kFloat datatype
        var yHat = [Int]()
        
        for i in 0..<testData.count {
            guard let outputs = module.predict(testData[i], withXDim: 1, andYDim: Int32(testData[0].count)) else {
                return
            }
            
            
            if (Double(truncating: outputs[0]) > 0.5) {
                yHat.append(1)
            } else {
                yHat.append(0)
            }
        }
        
        var correct = 0
        for i in 0..<testLabels.count {
            if testLabels[i] == yHat[i] {
                correct = correct + 1
            }
        }
        
        print("Correct: \(correct), count: \(testData.count) Accuracy: \(Double(correct)/Double(testData.count))" )
        
        
        
        
        
        //        let zippedResults = zip(labels.indices, outputs)
        //        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
        //        var text = ""
        //        for result in sortedResults {
        //            text += "\u{2022} \(labels[result.0]) \n\n"
        //        }
        resultView.text = ""
    }
}
