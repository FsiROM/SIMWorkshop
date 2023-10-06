import numpy as np

def load_data(freqs, last_incr, last_tr_incr, start_incr, include_subiters):
    
    tr_disp_data = []
    tr_load_data = []
    test_disp_data = []
    test_load_data = []

    tr_first_iters = []
    tr_converged_iters = [] 
    flat_tr_converged_iters = [] 
    flat_tr_first_iters = [] 

    tst_first_iters = []
    tst_converged_iters = []
    flat_tst_converged_iters = [] 
    flat_tst_first_iters = [] 


    cutoff = last_tr_incr
    startoff = start_incr
    end_incr = last_incr
    
    folder_names = []
    for freq in freqs:
        folder_names.append("./trainingData/"+freq)

    num_of_params = len(folder_names)

    k = 0
    l = 0
    m = 0
    n = 0

    for i in range(num_of_params):

        loaded_iters_data = np.load(folder_names[i]+"/coSimData/iters.npy")
        startoff = start_incr
        if i == 0:
            startoff = 0
        start_id = np.append(0, loaded_iters_data.cumsum().astype(int))[startoff]

        iters = loaded_iters_data[startoff:cutoff]

        iters_tst = loaded_iters_data[cutoff:end_incr]

        tr_first_iters.append(np.append(0, iters[:-1].cumsum().astype(int))+start_id)
        tr_converged_iters.append((iters.cumsum()-1).astype(int)+start_id)

        flat_tr_converged_iters.append(tr_converged_iters[-1]+ k - start_id)
        flat_tr_first_iters.append(tr_first_iters[-1]+ l - start_id)

        tst_first_iters.append(np.append(0, iters_tst[:-1].cumsum().astype(int)))
        tst_converged_iters.append((iters_tst.cumsum()-1).astype(int))
        flat_tst_converged_iters.append(tst_converged_iters[-1] + m )
        flat_tst_first_iters.append(tst_first_iters[-1] + n)


        if include_subiters:
            tr_disp_data.append(np.load(folder_names[i]+"/coSimData/disp_data.npy")[:, start_id:tr_converged_iters[-1][-1]+1])
            tr_load_data.append(np.load(folder_names[i]+"/coSimData/load_data.npy")[:, start_id:tr_converged_iters[-1][-1]+1])

            test_disp_data.append(np.load(folder_names[i]+"/coSimData/disp_data.npy")[:, tr_converged_iters[-1][-1]+1:
                                                                                      tst_converged_iters[-1][-1]+1+tr_converged_iters[-1][-1]+1])
            test_load_data.append(np.load(folder_names[i]+"/coSimData/load_data.npy")[:, tr_converged_iters[-1][-1]+1:
                                                                                      tst_converged_iters[-1][-1]+1+tr_converged_iters[-1][-1]+1])

        else:            
            tr_disp_data.append(np.load(folder_names[i]+"/coSimData/disp_data.npy")[:, tr_converged_iters[-1]])
            tr_load_data.append(np.load(folder_names[i]+"/coSimData/load_data.npy")[:, tr_converged_iters[-1]])

            test_disp_data.append(np.load(folder_names[i]+"/coSimData/disp_data.npy")[:, tst_converged_iters[-1]+tr_converged_iters[-1][-1]+1])
            test_load_data.append(np.load(folder_names[i]+"/coSimData/load_data.npy")[:, tst_converged_iters[-1]+tr_converged_iters[-1][-1]+1])


        k += tr_converged_iters[-1][-1]+1-start_id
        l += tr_converged_iters[-1][-1]+1-start_id
        m += tst_converged_iters[-1][-1]+1
        n += tst_converged_iters[-1][-1]+1

    tr_disp_data = np.concatenate(tr_disp_data, axis=1)
    tr_load_data = np.concatenate(tr_load_data, axis = 1)
    test_disp_data = np.concatenate(test_disp_data, axis=1)
    test_load_data = np.concatenate(test_load_data, axis = 1)
    #flat_tr_converged_iters = np.concatenate((flat_tr_converged_iters))
    #flat_tr_first_iters = np.concatenate((flat_tr_first_iters))
    #flat_tst_converged_iters = np.concatenate((flat_tst_converged_iters))
    #flat_tst_first_iters = np.concatenate((flat_tst_first_iters))
    
    return tr_load_data, tr_disp_data, test_load_data, test_disp_data, flat_tr_converged_iters, flat_tr_first_iters, flat_tst_converged_iters, flat_tst_first_iters
