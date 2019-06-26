import numpy as np

frame0_train = np.loadtxt('/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame0_im.gz')

frame1_train = np.loadtxt('/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame1_im.gz')

frame0_label = np.loadtxt('/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame0_q.gz')

frame1_label = np.loadtxt('/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame1_q.gz')

combined_train = np.vstack((frame0_train,frame1_train))
combined_label = np.vstack((frame0_label,frame1_label))



for i in range(2,40):
    print(i)
    combined_train = np.vstack((combined_train,
                              np.loadtxt(f'/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame{i}_im.gz')))

    combined_label = np.vstack((combined_label,
                              np.loadtxt(f'/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame{i}_q.gz'))) 

    print(combined_train.shape)
    print(combined_label.shape)                      
                               
np.savetxt('combined_train.gz', combined_train)
np.savetxt('combined_label.gz', combined_label)
