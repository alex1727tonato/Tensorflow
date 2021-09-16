# Las utilidades requeridas

import numpy as np
np.set_printoptions(precision = 2)

def calc_table(C_in, C_print_flag=0):
    if(np.shape(C_in)==(7,7)):
        Cs5 = np.zeros((5,5))
        Cs5[1:,1:] = C_in[-4:,-4:]
        Cs5[0,1:] = np.sum(C_in[:3,3:], 0)
        Cs5[1:,0] = np.sum(C_in[3:,:3], 1)
        Cs5[0,0] = np.sum(C_in[:3,:3])
    else:
        Cs5=C_in

    if(C_print_flag==1):
        print(Cs5)
    if(C_print_flag==2):
        print(C_in)

    selected_class = 2
    TpV = Cs5[selected_class,selected_class]
    FpV = np.sum(Cs5[:,selected_class]) - TpV
    FnV = np.sum(Cs5[selected_class,:]) - TpV
    TnV = np.sum(Cs5[:,:]) - FpV - FnV - TpV
    accV = np.round(100*(TpV + TnV)/(TpV+TnV+FpV+FnV),2)
    senV = np.round(100*(TpV)/(TpV+FnV),2)
    speV = np.round(100*(TnV)/(TnV+FpV),2)
    pprV = np.round(100*(TpV)/(TpV+FpV),2)

    selected_class = 1
    TpS = Cs5[selected_class,selected_class]
    FpS = np.sum(Cs5[:,selected_class]) - TpS
    FnS = np.sum(Cs5[selected_class,:]) - TpS
    TnS = np.sum(Cs5[:,:]) - FpS - FnS - TpS
    accS = np.round(100*(TpS + TnS)/(TpS+TnS+FpS+FnS),2)
    senS = np.round(100*(TpS)/(TpS+FnS),2)
    speS = np.round(100*(TnS)/(TnS+FpS),2)
    pprS = np.round(100*(TpS)/(TpS+FpS),2)

    outputMat = np.reshape(np.asarray([accV, senV, speV, pprV, accS, senS, speS, pprS]), (1,-1))
    
    return outputMat


def calc_tables(allCs, n_classes):
    '''print("\nDear Chazal all!")
    ChazalIDs = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234'] #chazal DS2 test
    Cs=np.zeros((n_classes,n_classes))
    for Id in ChazalIDs:
        Cs = Cs + allCs[Id]
    tab = calc_table(Cs)

    print("\nDear 11 Common!")
    com11 = ['200', '202', '210', '213', '214', '219', '221', '228', '231', '233', '234']
    Cs=np.zeros((n_classes,n_classes))
    for Id in com11:
        Cs = Cs + allCs[Id]
	tab = calc_table(Cs)

    print("\nDear 14 Common!")
    com14 = ['200', '202', '210', '213', '214', '219', '221', '228', '231', '233', '234', '212', '222', '232']
    Cs=np.zeros((n_classes,n_classes))
    for Id in com14:
        Cs = Cs + allCs[Id]
    tab = calc_table(Cs)

    print("\nDear 100 etc!")
    allRecs = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124']
    Cs=np.zeros((n_classes,n_classes))
    for Id in allRecs:
        Cs = Cs + allCs[Id]
    tab = calc_table(Cs)'''
    
    #print("\nDear 200 onward!")
    up200 = ['200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
    Cs=np.zeros((n_classes,n_classes))
    for Id in up200:
        Cs = Cs + allCs[Id]
    tab = calc_table(Cs)

    return tab
