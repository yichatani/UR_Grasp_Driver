import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/artc/UR_Grasp_Driver/yc_ws/install/frames'
