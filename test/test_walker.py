from test_setup import *
import fiso.walker.walker3 as fww
import fiso.tools.contour as ftc
timer('init')
iso_dict,label,_,_ = fww.find(potential)
timer('finish')
ftc.plot_tree_outline(-potential,iso_dict)
ftc.p.show()
