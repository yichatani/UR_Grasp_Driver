from grasp_detector import GraspDetector
# from grasp_detector_2 import GraspPoseInference
class GraspDetectorFactory:
    @staticmethod
    def create_grasp_detector(choice, coordinator=None):
        if choice == 1:
            return GraspDetector(coordinator)
        # elif choice == 2:
        #     return GraspPoseInference(coordinator)
        else:
            raise ValueError(f"Invalid choice: {choice}. Please select 1 for 'AnyGrasp' or 2 for 'Grasp Pose Inference'.")