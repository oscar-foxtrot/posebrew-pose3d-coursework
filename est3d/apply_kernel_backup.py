def get_kpts(file_number):

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    file_path_0 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_0/X3D.npy'
    kpts0 = np.load(file_path_0)
    
    file_path_1 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_1/X3D.npy'
    if not os.path.exists(file_path_1):
        kpts1 = None
    else:
        kpts1 = np.load(file_path_1)
    
    file_path_2 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_2/X3D.npy'
    if not os.path.exists(file_path_2):
        kpts2 = None
    else:
        kpts2 = np.load(file_path_2)

    # Plots
    '''
    m_frame = 200
    errors = kpts1 - kpts0[81:]
    frame_diff = kpts1[1:] - kpts1[:-1]
    forearm_lenth = ((kpts1[:, 12, 0] - kpts1[:, 13, 0])**2 + (kpts1[:, 12, 1] - kpts1[:, 13, 1])**2 + (kpts1[:, 12, 2] - kpts1[:, 13, 2])**2)**0.5
    dist_diff = np.linalg.norm(frame_diff, axis=2)
    print(dist_diff.shape)
    # See the magnitude of errors
    forearm_lenth = ((kpts1[:, 12, 0] - kpts1[:, 13, 0])**2 + (kpts1[:, 12, 1] - kpts1[:, 13, 1])**2 + (kpts1[:, 12, 2] - kpts1[:, 13, 2])**2)**0.5
    #print(np.max(errors / forearm_lenth[:, np.newaxis, np.newaxis], axis=0))

    '''
    '''
    print(kpts0[m_frame + 81])
    print(kpts1[m_frame])
    print(errors / (kpts1[m_frame] - kpts1[m_frame + 1]))
    print(np.max(errors / (kpts1[m_frame] - kpts1[m_frame + 1])))
    print(np.average(errors / (kpts1[m_frame] - kpts1[m_frame + 1])))


    print(np.average((kpts1[m_frame] - kpts0[m_frame + 81]) / (kpts1[m_frame] - kpts1[m_frame + 1])))
    '''

    '''
    print(kpts0[560 + 162])
    print(kpts1[560 + 81])
    print(kpts2[560])
    '''
    '''
    #plt.plot(range(1209), np.average(errors / forearm_lenth[:, np.newaxis, np.newaxis], axis=(1,2)), label='average interframe joint position error\n(normalized by the forearm length)')
    #plt.scatter(range(0, 1209, 81), np.average(errors / forearm_lenth[:, np.newaxis, np.newaxis], axis=(1,2))[::81], color='red', s=30, label='')
    plt.plot(range(1, 1209), np.average(dist_diff, axis=1) / forearm_lenth[1:],
        label='Средняя разница в позиции точки на теле\nмежду кадрами i и i - 1\n(нормализованная на длину предплечья)')
    pts = [i for i in range(1, 1209) if (i + 1) % 243 == 0]
    plt.scatter(pts, np.average(dist_diff, axis=1)[pts] / forearm_lenth[pts],
        color='red', s=30, label='Точки, соответствующие кадрам с номерами\ni: i mod 243 = 0\n'
            '(Аномальное поведение в конце обусловлено в\nтом числе ошибками детекции вследствие того,\nчто субъект находится слишком близко к камере)')

    print(np.average(dist_diff, axis=1)[pts] / forearm_lenth[pts])
    plt.legend()
    #plt.savefig("error_profile_rus.png", dpi=300)
    plt.show()
    '''

    def kernel(x, sigma=40, center=121):
        return np.exp(-1 / 2 * ((x % 243 - center) / sigma)**2)


    # It is expected for kpts0 to always exist
    kpts = []
    for i in range(kpts0.shape[0]):
        res = 0
        wghts = 0

        wghts += kernel(i, center=121)
        res += kpts0[i] * kernel(i, center=121)

        if i >= 81 and kpts1 is not None:
            wghts += kernel(i - 81, center=121)
            res += kpts1[i - 81] * kernel(i - 81, center=121)

        if i >= 162 and kpts2 is not None:
            wghts += kernel(i - 162, center=121)
            res += kpts2[i - 162] * kernel(i - 162, center=121)

        res /= wghts
        kpts += [res]
    
    return np.array(kpts)


    '''
    kpts = [(kpts0[i] * kernel(i, center=121) +
        kpts1[i] * kernel(i, center=121 + 81) +
        kpts2[i] * kernel(i, center=121 + 81 * 2)) /
        kernel(i, center=121) + kernel(i, center=121 + 81) + kernel(i, center=121 + 81 * 2)
        for i in range(kpts.shape[0])]
    '''
    '''
    m = 121 + 162 + 20
    print(kernel(m, center=121))
    print(kernel(m - 81, center=121))
    print(kernel(m - 162, center=121))
    '''