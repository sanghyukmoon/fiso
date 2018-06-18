from test_setup import *
import mem
import fiso
import fiso_beta as fb
import code_cf_b as cf
verbose = False
fiso.verbose = verbose
fb.verbose = verbose
cf.verbose = verbose
timer('init')
_dict,fiso_labels = fiso.find(potential[::2,::2,::2])
timer('fiso')
_dict,fb_labels = fb.find(potential[::2,::2,::2])
timer('beta')
_dict,cf_labels = cf.corefind(potential[::2,::2,::2])
timer('corefind')

cffiso = (cf_labels != fiso_labels)
cfbeta = (cf_labels != fb_labels)
if n.any(cffiso):
    print(n.sum(cffiso),'wrong')
if n.any(cfbeta):
    print(n.sum(cfbeta),'wrong')


timer('init')
fiso.find(potential)
timer('fiso')
fb.find(potential)
timer('beta')
