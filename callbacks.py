from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path):
        super(CustomCallback, self).__init__()
        self.model_save_path = model_save_path
        self.best_val_loss = np.inf
        self.patience = 3
        self.reduce_lr_factor = 0.5

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')

        # Kiểm tra xem val_loss có cải thiện hay không
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save(self.model_save_path)  # Lưu mô hình khi val_loss cải thiện

        # Kiểm tra xem val_loss đã không cải thiện trong 'patience' epochs liên tiếp hay không
        if epoch > self.patience - 1:
            prev_losses = [logs.get(f'val_loss_{i}') for i in range(1, self.patience + 1)]  # Sửa đổi ở đây
            if all(loss >= self.best_val_loss for loss in prev_losses):
                # Giảm learning rate nếu val_loss không cải thiện trong 'patience' epochs liên tiếp
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = lr * self.reduce_lr_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'Reducing learning rate to: {new_lr}')

# Đặt tên file model để lưu
model_save_path = 'best_model.h5'
# Tạo callback tổng hợp
custom_callback = CustomCallback(model_save_path)

# Tạo callback ModelCheckpoint để lưu model
model_checkpoint_callback = ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,  # Chỉ lưu model tốt nhất
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Tạo callback ReduceLROnPlateau để giảm learning rate khi val_loss không cải thiện
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=custom_callback.patience,  # Sửa đổi ở đây
    verbose=1,
    mode='min'
)

