import cv2
import preprocess

if __name__=='__main__':
    file_paths=preprocess.get_file_path('../Solar_Panel_Soiling_Image_dataset/PanelImages')
    import ipdb; ipdb.set_trace()

    L, I=preprocess.get_li(file_paths)

    print(L[0], I[0], file_paths[0])
