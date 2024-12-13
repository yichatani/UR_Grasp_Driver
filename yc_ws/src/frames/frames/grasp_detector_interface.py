from abc import ABC, abstractmethod

class GraspDetectorInterface(ABC):


    @abstractmethod
    def detect_grasps(self, color_image, depth_image):

        pass

    def set_coordinator(self, coordinator):
        self.coordinator = coordinator
