path = 'datasets/'
file = open(path + "simstudy_norm.txt", "r")
data = file.readlines()
file.close() 

w = 'Wald'
w_bs = 'BS Wald'
w_bst = 'BS-t Wald'
lr = 'Likelihood-ratio'
lr_bs = 'BS Likelihood-ratio'

bonf = 'Bonf'
ks = 'KS-Test'
ks_bs = 'BS KS-Test'
asymp_delta = 'Delta-method'
w_delta = 'BS Delta-method'
studw_delta = 'BS-t Delta-method'
asymp_mc = 'Monte-carlo'
w_mc = 'BS Monte-carlo'
studw_mc = 'BS-t Monte-carlo'
asymp_nm = 'Nelder-mead'
w_nm = 'BS Nelder-mead'
studw_nm ='BS-t Nelder-mead'

for i,row in enumerate(data):
    if 3<i and i<43 and ( (i-4) % 10 == 0 ):
        w += ' & ' + row[:-1]
    if 3<i and i<43 and ( (i-4) % 10 == 2 ):
        w_bs += ' & ' + row[:-1]
    if 3<i and i<43 and ( (i-4) % 10 == 4 ):
        w_bst += ' & ' + row[:-1]
    if 3<i and i<43 and ( (i-4) % 10 == 6 ):
        lr += ' & ' + row[:-1]
    if 3<i and i<43 and ( (i-4) % 10 == 8 ):
        lr_bs += ' & ' + row[:-1]
    if i==43:
        print(w + '\\\\')
        print(w_bs + '\\\\')
        print(w_bst + '\\\\')
        print(lr + '\\\\')
        print(lr_bs + '\\\\')
        print('')
    if 46<i and i<142 and ( (i-47) % 24 == 0 ):
        bonf += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 2 ):
        ks += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 4 ):
        ks_bs += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 6 ):
        asymp_delta += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 8 ):
        w_delta += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 10 ):
        studw_delta += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 12 ):
        asymp_mc += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 14 ):
        w_mc += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 16 ):
        studw_mc += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 18 ):
        asymp_nm += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 20 ):
        w_nm += ' & ' + row[:-1]
    if 46<i and i<142 and ( (i-47) % 24 == 22 ):
        studw_nm += ' & ' + row[:-1]

print(bonf + '\\\\')
print(ks + '\\\\')
print(ks_bs + '\\\\')
print(asymp_delta + '\\\\')
print(w_delta + '\\\\')
print(studw_delta + '\\\\')
print(asymp_mc + '\\\\')
print(w_mc + '\\\\')
print(studw_mc + '\\\\')
print(asymp_nm + '\\\\')
print(w_nm + '\\\\')
print(studw_nm + '\\\\')
print('')