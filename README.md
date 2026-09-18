Single_Lead_ECG_AF_Classification/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/                               # [در گیت Ignore شود] کل داده‌ها در یک جا
│   ├── raw/                            # داده‌های خام PhysioNet (فولدرهای A00 تا A08)
│   ├── processed/                      # داده‌های پیش‌پردازش‌شده (dl_data)
│   └── splits/                         # تقسیم‌بندی Train/Val/Test یا K-Fold
│
├── notebooks/                          # فقط نوت‌بوک‌های EDA و تحلیل اولیه
│   ├── 01_raw_data_eda.ipynb
│   ├── 02_ecg_filtering_and_features.ipynb
│   └── 03_split_distributions.ipynb
│
├── src/                                # کدهای مشترک و ماژولار (Core Logic)
│   ├── __init__.py
│   ├── config.py                       # مسیرها، هایپرپارامترها، Sampling Rate، طول سیگنال و Seed
│   ├── dataset.py                      # کلاس Dataset و DataLoader اختصاصی PyTorch
│   ├── preprocessing.py                # توابع فیلتر Bandpass، حذف Baseline Wander، نرمال‌سازی
│   ├── metrics.py                      # محاسبه F1 چالش (F1-Normal, F1-AF, F1-Other, F1-Avg) و Confusion Matrix
│   ├── utils.py                        # توابع ذخیره وزن، لاگ، رسم نمودار
│   │
│   └── models/                         # تمام معماری‌ها در یک جا (بدون تکرار کد لود دیتا)
│       ├── __init__.py
│       ├── cnn_1d.py                   # مدل 01_baseline_1D_CNN
│       ├── resnet_1d.py                # مدل 02_ResNet_1D
│       ├── inception_time.py           # مدل 03_InceptionTime_1D
│       ├── xception_1d.py              # مدل 04_Xception_1D
│       ├── densenet_1d.py              # مدل 05_DenseNet_1D
│       ├── rnn_bilstm.py               # مدل 08_BiLstm_1D
│       ├── transformers.py             # مدل‌های 06 و 09 (Transformer_1D)
│       ├── hybrid_cnn_transformer.py   # مدل 07_Hybrid
│       └── proposed_model.py           # مدل اصلی و نهایی خودتان
│
├── scripts/                            # اسکریپت‌های اجرایی
│   ├── prepare_data.py                 # تبدیل دیتای خام به processed و ساخت splits
│   ├── train.py                        # اسکریپت عمومی آموزش (با سوییچ نام مدل، مثلا: --model xception)
│   └── evaluate.py                     # مقایسه همه مدل‌ها روی تست‌ست و تولید جداول گزارش
│
└── experiments/                        # [در گیت Ignore شود] خروجی اجراها
    ├── checkpoints/                    # فایل‌های pth بهترین مدل‌ها (مثلا best_xception.pth)
    ├── logs/                           # لاگ‌های آموزش یا خروجی‌های WandB / TensorBoard
    └── results/                        # ماتریس‌های درهم‌ریختگی (Confusion Matrices) و نمودارهای Loss/F1
