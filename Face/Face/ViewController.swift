import UIKit
import Vision

final class ViewController: UIViewController {
    let faceDetection = VNDetectFaceRectanglesRequest()
    let faceLandmarks = VNDetectFaceLandmarksRequest()
    
    var image1Points: [(x: CGFloat, y: CGFloat)] = []
    var image2Points: [(x: CGFloat, y: CGFloat)] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        configureRequests()
        
        let titleLabel = UILabel()
        titleLabel.text = "FACE RECOGNITION"
        titleLabel.textAlignment = .center
        titleLabel.font = UIFont.systemFont(ofSize: 25, weight: .bold)
        titleLabel.frame = CGRect(x: 0, y: 100, width: view.bounds.width, height: 30)
        view.addSubview(titleLabel)
        
        guard let image1 = UIImage(named: "00783"), let image2 = UIImage(named: "00783") else {
            print("Image not found or failed to convert UIImage to CGImage")
            return
        }
        
        let framedsize = CGSize(width: 750, height: 750)
        let resizedImage1 = image1.resized(to: framedsize)
        let resizedImage2 = image2.resized(to: framedsize)

        
        print("Image 1 size: \(resizedImage1.size).\n")
        print("Image 2 size: \(resizedImage2.size).\n")
        
        guard let cgImage1 = resizedImage1.cgImage, let cgImage2 = resizedImage2.cgImage else {
            print("Failed to convert UIImage to CGImage")
            return
        }
        
        let ciImage1 = CIImage(cgImage: cgImage1).oriented(forExifOrientation: Int32(UIImage.Orientation.up.rawValue))
        let ciImage2 = CIImage(cgImage: cgImage2).oriented(forExifOrientation: Int32(UIImage.Orientation.up.rawValue))
        
        DispatchQueue.main.async {
            self.detectFace(on: ciImage1, imageIndex: 1)
            self.detectFace(on: ciImage2, imageIndex: 2)
        }
        
        let imageView1 = UIImageView(image: resizedImage1)
        imageView1.contentMode = .scaleAspectFit
        imageView1.frame = CGRect(x: 0, y: 0, width: view.bounds.width / 2, height: view.bounds.height)
        view.addSubview(imageView1)
        
        let imageView2 = UIImageView(image: resizedImage2)
        imageView2.contentMode = .scaleAspectFit
        imageView2.frame = CGRect(x: view.bounds.width / 2, y: 0 , width: view.bounds.width / 2, height: view.bounds.height)
        view.addSubview(imageView2)
        
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
    }

    func configureRequests() {
        #if targetEnvironment(simulator)
            if #available(iOS 17.0, *) {
                let allDevices = MLComputeDevice.allComputeDevices
                for device in allDevices {
                    if(device.description.contains("MLCPUComputeDevice")){
                        faceDetection.setComputeDevice(.some(device), for: .main)
                        faceLandmarks.setComputeDevice(.some(device), for: .main)
                        break
                    }
                }
            } else {
                faceDetection.usesCPUOnly = true
                faceLandmarks.usesCPUOnly = true
            }
        #endif
    }

    func detectFace(on image: CIImage, imageIndex: Int) {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        do {
            try handler.perform([faceDetection])
            print("Face detection request performed on image \(imageIndex).\n")
        } catch {
            print("Face detection failed on image \(imageIndex): \(error)")
            return
        }
        
        guard let results = faceDetection.results, !results.isEmpty else {
            print("No face detected on image \(imageIndex).")
            return
        }
        
        print("Face detected on image \(imageIndex): \(results.count)\n")
        faceLandmarks.inputFaceObservations = results
        detectLandmarks(on: image, imageIndex: imageIndex)
    }

    func detectLandmarks(on image: CIImage, imageIndex: Int) {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        do {
            try handler.perform([faceLandmarks])
            print("Face landmarks detection request performed on image \(imageIndex).\n")
        } catch {
            print("Face landmarks detection failed on image \(imageIndex): \(error)")
            return
        }
        
        guard let landmarksResults = faceLandmarks.results, !landmarksResults.isEmpty else {
            print("No landmarks detected on image \(imageIndex).")
            return
        }
        
        print("Landmarks detected on image \(imageIndex): \(landmarksResults.count)\n")
        
        for observation in landmarksResults {
            if let boundingBox = self.faceLandmarks.inputFaceObservations?.first?.boundingBox {
                let faceBoundingBox = boundingBox.scaled(to: self.view.bounds.size)
                print("BoundingBox on image \(imageIndex): \(faceBoundingBox)\n")
                if let allpoints = observation.landmarks?.allPoints {
                    print("All points detected on image \(imageIndex).\n")
                    let points = self.convertPointsForFace(allpoints, faceBoundingBox)
                    print("Points detected: \n\(points)\n")
                    if imageIndex == 1 {
                        self.image1Points = points
                    } else if imageIndex == 2 {
                        self.image2Points = points
                    }
                    if !self.image1Points.isEmpty && !self.image2Points.isEmpty {
                        self.calculateDistance()
                    }
                }
            }
        }
    }

    func convertPointsForFace(_ landmark: VNFaceLandmarkRegion2D?, _ boundingBox: CGRect) -> [(x: CGFloat, y: CGFloat)]{
        guard let landmark = landmark else { return [] }
        let points = landmark.normalizedPoints
        let count = landmark.pointCount
        
        let convertedPoints = convert(points, with: count)
        
        return convertedPoints.map { point in
            let pointX = (point.x * boundingBox.width) + boundingBox.origin.x
            let pointY = (point.y * boundingBox.height) + boundingBox.origin.y
            return (x: pointX, y: pointY)
        }
    }
    
    
    func convert(_ points: UnsafePointer<CGPoint>, with count: Int) -> [(x: CGFloat, y: CGFloat)] {
        var convertedPoints = [(x: CGFloat, y: CGFloat)]()
        for i in 0..<count {
            convertedPoints.append((CGFloat(points[i].x), CGFloat(points[i].y)))
        }
        return convertedPoints
    }
    

    func calculateDistance() {
        var totalDistance: CGFloat = 0.0
        print("Image 1 Points: \(image1Points.count)")
        print("Image 2 Points: \(image2Points.count)\n")
        for i in 0..<image1Points.count {
            let point1 = image1Points[i]
            let point2 = image2Points[i]
            let distance = hypot(point1.x - point2.x, point1.y - point2.y)
            totalDistance += distance
            print("Point \(i) Distance: \(distance)\n")
        }
        let averageDistance = totalDistance / CGFloat(image1Points.count)
        print("Average Distance: \(averageDistance)\n")
        
        let threshold: CGFloat = 0.13
        if (averageDistance < threshold) {
            print("Same Person!\n")
        } else {
            print("Different Person!\n")
        }
    }
}
