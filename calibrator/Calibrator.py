import json
import numpy as np
from scipy.special import wofz
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def Lorentzian(x: np.ndarray, center: float, intensity: float, w: float, bg: float = 0) -> np.ndarray:
    y = w ** 2 / (4 * (x - center) ** 2 + w ** 2)
    return intensity * y + bg


def Gaussian(x: np.ndarray, center: float, intensity: float, sigma: float, bg: float = 0) -> np.ndarray:
    y = np.exp(-1 / 2 * (x - center) ** 2 / sigma ** 2)
    return intensity * y + bg


def Voigt(x: np.ndarray, center: float, intensity: float, lw: float, gw: float, bg: float = 0) -> np.ndarray:
    # lw : FWHM of Lorentzian
    # gw : sigma of Gaussian
    if gw == 0:
        gw = 1e-10
    z = (x - center + 1j*lw) / (gw * np.sqrt(2.0))
    w = wofz(z)
    model_y = w.real / (gw * np.sqrt(2.0*np.pi))
    intensity /= model_y.max()
    return intensity * model_y + bg


class Calibrator:
    def __init__(self, material: str = None, dimension: int = None, measurement: str = None, xdata: np.ndarray = None, ydata: np.ndarray = None):
        self.measurement: str = measurement
        self.material: str = material
        self.dimension: int = dimension

        self.xdata_before: np.ndarray = None
        self.xdata: np.ndarray = xdata
        self.ydata: np.ndarray = ydata

        self.database = {
            "Raman": {
                "link": "https://www.chem.ualberta.ca/~mccreery/ramanmaterials.html",
                "sulfur": [85.1, 153.8, 219.1, 473.2],
                "naphthalene": [513.8, 763.8, 1021.6, 1147.2, 1382.2, 1464.5, 1576.6, 3056.4],
                "acetonitrile": [2253.7, 2940.8],
                "1,4-Bis(2-methylstyryl)benzene": [1177.7, 1290.7, 1316.9, 1334.5, 1555.2, 1593.1, 1627.9],
                "cyclohexane": [801.3, 1028.3, 1157.6, 1266.4, 1444.4, 2664.4, 2852.9, 2923.8, 2938.3]
            },
            "Rayleigh": {
                "link": "https://www.nist.gov/pml/atomic-spectra-database",
                "ArHg": [435.8335, 546.0750, 576.9610, 579.0670, 696.5431, 706.7218, 714.7042, 727.2936, 738.3980, 750.3869, 751.4652, 763.5106, 772.3761, 794.8176, 800.6157, 801.4786, 810.3693, 811.5311, 826.4522, 840.8210, 842.4648, 852.1442, 912.2967, 922.4499, 965.7786, 978.4503, 1013.975, 1047.0054, 1067.3565, 1128.71, 1148.811, 1166.871, 1211.2326, 1270.228, 1280.274, 1295.666, 1300.826]
            },
            "PL": {
                "link": "https://www.nist.gov/pml/atomic-spectra-database",
                "ArHg": [435.8335, 546.0750, 576.9610, 579.0670, 696.5431, 706.7218, 714.7042, 727.2936, 738.3980, 750.3869, 751.4652, 763.5106, 772.3761, 794.8176, 800.6157, 801.4786, 810.3693, 811.5311, 826.4522, 840.8210, 842.4648, 852.1442, 912.2967, 922.4499, 965.7786, 978.4503, 1013.975, 1047.0054, 1067.3565, 1092.14, 1128.71, 1148.811, 1166.871, 1211.2326, 1270.228, 1280.274, 1295.666, 1300.826, 1393.1, 1527.02]
            }
        }

        self.functions = {
            'Lorentzian': Lorentzian,
            'Gaussian': Gaussian,
            'Voigt': Voigt
        }
        self.function = Lorentzian
        self.num_params = 4
        self.search_width = 10

        self.pf = PolynomialFeatures()
        self.lr = LinearRegression()

        self.fitted_x: np.ndarray = None
        self.found_x_true: np.ndarray = None
        self.is_calibrated: bool = False
        self.calibration_info = ['', 0, '', []]  # material, dimension, function, peak_positions

    def set_data(self, xdata: np.ndarray, ydata: np.ndarray):
        if len(xdata.shape) != 1 or len(ydata.shape) != 1:
            raise ValueError('Invalid shape. x and y array must be 1 dimensional.')
        if xdata.shape != ydata.shape:
            raise ValueError('Invalid shape. x and y array must have same shape.')
        self.xdata = xdata.copy()
        self.ydata = ydata.copy()

    def set_measurement(self, measurement: str):
        if measurement not in self.database.keys():
            raise ValueError(f'Invalid measurement. It must be {", or ".join(self.database.keys())}')
        self.measurement = measurement

    def set_material(self, material: str):
        if material not in self.database[self.measurement].keys():
            raise ValueError(f'Invalid material. It must be {", or ".join(self.database[self.measurement].keys())}')
        self.material = material

    def set_dimension(self, dimension: int):
        if dimension < 0:
            raise ValueError('Invalid dimension. It must be greater than zero.')
        self.dimension = dimension

    def set_function(self, function: str):
        if function not in self.functions.keys():
            raise ValueError(f'Invalid function. It must be {", or ".join(self.functions.keys())}')
        if function in ['Lorentzian', 'Gaussian']:
            self.num_params = 4
        elif function in ['Voigt']:
            self.num_params = 6
        else:
            raise ValueError('Unrecognised function.')
        self.function = self.functions[function]

    def set_search_width(self, search_width: float):
        self.search_width = search_width

    def get_measurement_list(self):
        return list(self.database.keys())

    def get_material_list(self):
        return list(self.database[self.measurement].keys())[1:]

    def get_dimension_list(self):
        return ['1 (Linear)', '2 (Quadratic)', '3 (Cubic)']

    def get_function_list(self):
        return list(self.functions.keys())

    def get_true_x(self):
        return np.array(self.database[self.measurement][self.material])

    def _find_peaks(self) -> bool:
        x_true = self.get_true_x()
        x_true = x_true[(x_true > self.xdata.min()) & (x_true < self.xdata.max())]  # crop
        search_ranges = [[x-self.search_width, x+self.search_width] for x in x_true]

        fitted_x = []
        found_x_true = []
        for x_ref_true, search_range in zip(x_true, search_ranges):
            # Crop
            partial = (search_range[0] < self.xdata) & (self.xdata < search_range[1])
            x_partial = self.xdata[partial]
            y_partial = self.ydata[partial]

            # Begin with finding the maximum position
            found_peaks, properties = find_peaks(y_partial, prominence=50)
            if len(found_peaks) == 0:
                print(f'Peak {x_ref_true} not detected.')
                continue
            if len(found_peaks) > 1:
                print(f'Many peaks around {x_ref_true}.')
                continue

            if self.num_params == 4:
                p0 = [x_partial[found_peaks[0]], y_partial[found_peaks[0]], 1, y_partial.min()]
            elif self.num_params == 6:
                p0 = [x_partial[found_peaks[0]], y_partial[found_peaks[0]], 3, 3, y_partial.min()]

            popt, pcov = curve_fit(self.function, x_partial, y_partial, p0=p0)

            fitted_x.append(popt[0])
            found_x_true.append(x_ref_true)

        # if no peak found or if only one peak found
        if len(fitted_x) < 2:  # reshape will be failed if there is only one peak
            return False

        self.fitted_x = np.array(fitted_x)
        self.found_x_true = np.array(found_x_true)
        return True

    def _find_peaks_easy(self) -> bool:
        x_true = self.get_true_x()
        x_true = x_true[(x_true > self.xdata.min()) & (x_true < self.xdata.max())]  # crop
        search_ranges = [[x-self.search_width, x+self.search_width] for x in x_true]

        fitted_x = []
        found_x_true = []
        for x_ref_true, search_range in zip(x_true, search_ranges):
            # Crop
            partial = (search_range[0] < self.xdata) & (self.xdata < search_range[1])
            x_partial = self.xdata[partial]
            y_partial = self.ydata[partial]

            max_pos = x_partial[y_partial == y_partial.max()][0]
            fitted_x.append(max_pos)
            found_x_true.append(x_ref_true)

        # if no peak found or if only one peak found
        if len(fitted_x) < 2:  # reshape will be failed if there is only one peak
            return False

        self.fitted_x = np.array(fitted_x)
        self.found_x_true = np.array(found_x_true)
        return True

    def _find_peaks_manually(self, ranges, x_true, curvefit=True) -> bool:
        # ranges: list of tuple (x0, y0, x1, y1)
        fitted_x = []
        found_x_true = []
        for (x0, y0, x1, y1), xt in zip(ranges, x_true):
            # Crop
            partial_x = (x0 < self.xdata) & (self.xdata < x1)
            partial_y = (y0 < self.ydata) & (self.ydata < y1)
            partial = partial_x & partial_y
            x_partial = self.xdata[partial]
            y_partial = self.ydata[partial]

            # Begin with finding the maximum position
            found_peaks, properties = find_peaks(y_partial, prominence=100)
            if len(found_peaks) == 0:
                print(f'Peak {xt} not detected.')
                continue
            if len(found_peaks) > 1:
                print(f'Many peaks around {xt}.')
                continue

            if self.num_params == 4:
                p0 = [x_partial[found_peaks[0]], y_partial[found_peaks[0]], 1, y_partial.min()]
            elif self.num_params == 6:
                p0 = [x_partial[found_peaks[0]], y_partial[found_peaks[0]], 3, 3, y_partial.min()]

            # PLではカーブフィットがうまくいかないので、ピーク位置をそのまま使う
            # レイリーやラマンでは大丈夫みたいなので、ピーク位置を参考に関数をカーブフィットする
            if curvefit:
                popt, pcov = curve_fit(self.function, x_partial, y_partial, p0=p0)
                fitted_x.append(popt[0])
            else:
                fitted_x.append(x_partial[found_peaks[0]])

            found_x_true.append(xt)

        # if no peak found or if only one peak found
        if len(fitted_x) < 2:  # reshape will be failed if there is only one peak
            return False

        self.fitted_x = np.array(fitted_x)
        self.found_x_true = np.array(found_x_true)
        return True

    def _train(self) -> None:
        self.pf.set_params(degree=self.dimension)
        fitted_x_poly = self.pf.fit_transform(self.fitted_x.reshape(-1, 1))

        # Train the linear model
        self.lr.fit(fitted_x_poly, np.array(self.found_x_true).reshape(-1, 1))

    def calibrate(self, mode: str = '', ranges=None, x_true: list = None, curvefit: bool = True) -> bool:
        if self.measurement is None or self.material is None or self.dimension is None or self.xdata is None or self.ydata is None:
            raise ValueError('Set up is not completed.')
        if mode not in ['', 'easy', 'manual']:
            raise ValueError('Invalid mode. It must be "", "easy", or "manual".')

        if mode == 'easy':
            ok = self._find_peaks_easy()
            if not ok:
                return False
            self.calibration_info = [self.material, self.dimension, 'easy', self.found_x_true.tolist()]
        elif mode == 'manual':
            ok = self._find_peaks_manually(ranges, x_true, curvefit)
            if not ok:
                return False
            self.calibration_info = [self.material, self.dimension, self.function.__name__, self.found_x_true.tolist()]
        else:
            ok = self._find_peaks()
            if not ok:
                return False
            self.calibration_info = [self.material, self.dimension, self.function.__name__, self.found_x_true.tolist()]

        self._train()

        self.xdata_before = self.xdata.copy()
        x = self.pf.fit_transform(self.xdata.reshape(-1, 1))
        self.xdata = np.ravel(self.lr.predict(x))

        self.is_calibrated = True

        return True

    def show_fit_result(self, ax) -> None:
        # 標準サンプルのピークのフィッティングの結果を表示
        # 較正前のピーク位置、較正後のピーク位置を示す
        if self.fitted_x is None or self.found_x_true is None:
            raise ValueError('Calibration is not performed yet.')
        ax.plot(self.xdata_before, self.ydata, color='k', linewidth=1, linestyle='dashed', label='Before')
        ax.plot(self.xdata, self.ydata, color='k', linewidth=1, label='After')
        ymin, ymax = ax.get_ylim()

        for i, (fitted_x, true_x) in enumerate(zip(self.fitted_x, self.found_x_true)):
            if i == 0:
                ax.vlines(fitted_x, ymin, ymax, color='r', linewidth=1, label='Found peak')
                ax.vlines(true_x, ymin, ymax, color='b', linewidth=1, label='True value')
            else:
                ax.vlines(fitted_x, ymin, ymax, color='r', linewidth=1)
                ax.vlines(true_x, ymin, ymax, color='b', linewidth=1)
            ax.text(fitted_x, ymax*0.95, str(round(fitted_x, 2)), color='r', fontsize=15)
            ax.text(true_x, ymax, str(round(true_x, 2)), color='b', fontsize=15)
        ax.legend(fontsize=15)
