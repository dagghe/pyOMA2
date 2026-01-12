# Interactive Stability Chart với Spectrum Overlay

## 🎯 Tính năng mới

Phiên bản này thêm tính năng **interactive mode** vào phương thức `plot_stab()`, cho phép:
- ✅ Pick modes trực tiếp trên stability chart
- ✅ Hiển thị spectrum overlay (CMIF plot)
- ✅ Toggle spectrum on/off trong lúc picking
- ✅ Giao diện tương tác với Tkinter GUI

## 📚 API Mới

### SSI Algorithm

```python
from pyoma2.algorithms import SSI
from pyoma2.setup import SingleSetup

# Khởi tạo và chạy SSI
ssicov = SSI(name="SSIcov", method="cov", br=50, ordmax=60, step=2)
setup = SingleSetup(data, fs=100)
setup.add_algorithms(ssicov)
setup.run_by_name("SSIcov")

# CŨ: Hai phương thức riêng biệt
fig, ax = ssicov.plot_stab(spectrum=True)  # Chỉ xem
ssicov.mpe_from_plot()                      # Interactive nhưng không có spectrum

# MỚI: Tích hợp cả hai!
ssicov.plot_stab(
    freqlim=(0, 50),
    spectrum=True,      # Hiển thị CMIF overlay
    interactive=True,   # Bật chế độ tương tác
    nSv=3,             # Số singular values
    hide_poles=True    # Ẩn unstable poles
)
```

### pLSCF Algorithm

```python
from pyoma2.algorithms import pLSCF

plscf = pLSCF(name="polymax", ordmax=40)
setup.add_algorithms(plscf)
setup.run_by_name("polymax")

# Tương tự như SSI
plscf.plot_stab(
    freqlim=(0, 50),
    spectrum=True,      # MỚI: Spectrum overlay
    interactive=True,   # MỚI: Interactive mode
    nSv="all"
)
```

## 🎮 Hướng dẫn sử dụng

### Điều khiển chuột

| Thao tác | Chức năng |
|----------|-----------|
| `SHIFT + Left Click` | Chọn pole gần nhất |
| `SHIFT + Right Click` | Xóa pole cuối cùng vừa chọn |
| `SHIFT + Middle Click` | Xóa pole gần con trỏ nhất |

### Menu

**File Menu:**
- `Save figure`: Lưu hình ảnh vào thư mục `pole_figures/`

**Show/Hide Unstable Poles Menu:**
- `Show unstable poles`: Hiển thị tất cả poles
- `Hide unstable poles`: Chỉ hiển thị stable poles

**Spectrum Overlay Menu** (Khi `spectrum=True`):
- `Show spectrum`: Hiển thị CMIF plot
- `Hide spectrum`: Ẩn CMIF plot

**Help Menu:**
- `Help`: Hiển thị hướng dẫn

## 📊 Ví dụ hoàn chỉnh

```python
import numpy as np
from pyoma2.algorithms import SSI
from pyoma2.setup import SingleSetup

# Load dữ liệu
data = np.load("measurement_data.npy")
setup = SingleSetup(data, fs=100)

# Khởi tạo SSI
ssicov = SSI(
    name="SSIcov",
    method="cov",
    br=50,
    ordmax=60,
    ordmin=2,
    step=2,
    calc_unc=True  # Tính uncertainty
)

setup.add_algorithms(ssicov)
setup.run_by_name("SSIcov")

# Mở GUI tương tác với spectrum
ssicov.plot_stab(
    freqlim=(1, 10),      # Giới hạn tần số
    spectrum=True,        # Hiển thị CMIF
    interactive=True,     # Interactive mode
    nSv=3,               # Top 3 singular values
    hide_poles=True,     # Ẩn unstable poles
    color_scheme="default"  # Colorblind-friendly
)

# Sau khi đóng GUI, các modes đã chọn được lưu
# Có thể tiếp tục extract modal parameters
print("Selected frequencies:", ssicov.result.Fn)
```

## 🆚 So sánh với API cũ

| Tính năng | API Cũ | API Mới |
|-----------|---------|---------|
| Xem stability chart | `plot_stab()` | `plot_stab()` |
| Spectrum overlay | `plot_stab(spectrum=True)` | `plot_stab(spectrum=True)` |
| Interactive picking | `mpe_from_plot()` | `plot_stab(interactive=True)` |
| Spectrum + Interactive | ❌ Không hỗ trợ | ✅ `plot_stab(spectrum=True, interactive=True)` |

## 🎨 Color Schemes

```python
# 4 color schemes có sẵn
ssicov.plot_stab(
    interactive=True,
    spectrum=True,
    color_scheme="default"  # or "classic", "high_contrast", "viridis"
)
```

- `default`: Colorblind-friendly (blue/orange)
- `classic`: Traditional (blue/orange)
- `high_contrast`: Black & white printing (black/gray)
- `viridis`: Purple/yellow gradient

## 🔧 Parameters

### `plot_stab()` - Full Signature

```python
def plot_stab(
    self,
    freqlim: Optional[tuple[float, float]] = None,
    hide_poles: bool = True,
    spectrum: bool = False,
    nSv: Union[int, "all"] = "all",
    color_scheme: Literal["default", "classic", "high_contrast", "viridis"] = "default",
    interactive: bool = False,
) -> tuple:
```

**Parameters:**
- `freqlim`: Frequency limits `(min, max)` in Hz
- `hide_poles`: Hide unstable poles for clarity
- `spectrum`: Enable CMIF overlay on secondary axis
- `nSv`: Number of singular values to display (int or "all")
- `color_scheme`: Color scheme for poles
- `interactive`: Enable interactive GUI for pole selection

**Returns:**
- `(fig, ax)`: Matplotlib Figure and Axes
- `(None, None)`: When `interactive=True` (GUI handles display)

## 🚀 Lợi ích

1. **Workflow đơn giản hơn**: 1 method thay vì 2
2. **Spectrum giúp identify modes**: So sánh trực tiếp với measured data
3. **Toggle real-time**: Bật/tắt spectrum trong lúc picking
4. **Consistent API**: SSI và pLSCF dùng cùng interface
5. **Backward compatible**: Không ảnh hưởng code cũ

## 📝 Notes

- Interactive mode yêu cầu Tkinter (thường có sẵn với Python)
- Spectrum overlay yêu cầu spectrum data (tự động estimate nếu chưa có)
- GUI chạy blocking - đợi user đóng window
- Selected modes không được extract modal parameters - cần gọi `mpe()` riêng nếu muốn extract

## 🐛 Troubleshooting

**Lỗi: "Tkinter not found"**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (với Homebrew Python)
brew install python-tk
```

**Lỗi: "est_spectrum() not found"**
- Đảm bảo algorithm có method `est_spectrum()`
- SSI/pLSCF có sẵn, FDD có thể cần implement

**GUI không hiển thị**
- Kiểm tra DISPLAY environment variable (Linux)
- Thử chạy từ terminal thay vì IDE

## 📖 Xem thêm

- Demo script: `Examples/interactive_stability_chart_demo.py`
- Documentation: `docs/docu/3_7 sel_from_plot module.rst`
- Source code: `src/pyoma2/support/sel_from_plot.py`
