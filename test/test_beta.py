from test_setup import *
import fiso
import fiso_beta as fb
timer('init')
core_dict,labels = fiso.find(potential[::2,::2,::2])
timer('fiso')
core_dict,labels = fb.find(potential[::2,::2,::2])
timer('beta')


timer('init')
core_dict,labels = fiso.find(potential)
timer('fiso')
core_dict,labels = fb.find(potential)
timer('beta')
