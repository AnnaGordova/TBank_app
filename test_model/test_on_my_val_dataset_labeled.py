from ultralytics import YOLO

def main():
    model = YOLO('../wheights/best.pt')


    results = model.val(
        data='../my_val_dataset_labeled/data.yaml',
        split='test',
    )

    metrics_df = {
        'Class': 'TBank',
        'Precision (IoU = 0.5)': results.box.mp,
        'Recall (IoU = 0.5)': results.box.mr,
        'F1-Score (IoU = 0.5)': results.box.f1[0],
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map
    }

    for k in metrics_df:
        print(f"{k} : {metrics_df[k]}")


if __name__ == '__main__':
    main()