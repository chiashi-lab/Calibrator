import numpy as np
from scipy.special import wofz
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def Lorentzian(x: np.ndarray, center: float, intensity: float, w: float) -> np.ndarray:
    y = w ** 2 / (4 * (x - center) ** 2 + w ** 2)
    return intensity * y


def Gaussian(x: np.ndarray, center: float, intensity: float, sigma: float) -> np.ndarray:
    y = np.exp(-1 / 2 * (x - center) ** 2 / sigma ** 2)
    return intensity * y


def Voigt(x: np.ndarray, center: float, intensity: float, lw: float, gw: float) -> np.ndarray:
    # lw : HWFM of Lorentzian
    # gw : sigma of Gaussian
    if gw == 0:
        gw = 1e-10
    z = (x - center + 1j*lw) / (gw * np.sqrt(2.0))
    w = wofz(z)
    model_y = w.real / (gw * np.sqrt(2.0*np.pi))
    intensity /= model_y.max()
    return intensity * model_y



class Calibrator:
    def __init__(self, material: str = None, dimension: int = None, measurement: str = None, xdata: np.ndarray = None, ydata: np.ndarray = None):
        self.measurement: str = measurement
        self.material: str = material
        self.dimension: int = dimension

        self.xdata: np.ndarray = xdata
        self.ydata: np.ndarray = ydata

        self.database = {
            "Raman": {
                "link": "https://www.chem.ualberta.ca/~mccreery/ramanmaterials.html",
                "sulfur": [85.1, 153.8, 219.1, 473.2],
                "naphthalene": [513.8, 763.8, 1021.6, 1147.2, 1382.2, 1464.5, 1576.6, 3056.4],
                "acetonitrile": [2253.7, 2940.8],
                "1,4-Bis(2-methylstyryl)benzene": [1177.7, 1290.7, 1316.9, 1334.5, 1555.2, 1593.1, 1627.9]
            },
            "Rayleigh": {
                "link": "https://www.nist.gov/pml/atomic-spectra-database",
                "ArHg": [435.8335, 546.0750, 576.9610, 579.0670, 696.5431, 706.7218, 714.7042, 727.2936, 738.3980, 750.3869, 751.4652, 763.5106, 772.3761, 794.8176, 800.6157, 801.4786, 810.3693, 811.5311]
            }
        }

        self.functions = {
            'Lorentzian': Lorentzian,
            'Gaussian': Gaussian,
            'Voigt': Voigt
        }
        self.function = Lorentzian

        self.pf = PolynomialFeatures()
        self.lr = LinearRegression()

        self.fitted_x = None
        self.found_x_true = None
        self.calibration_info = []

    def set_data(self, xdata: np.ndarray, ydata: np.ndarray):
        if len(self.xdata.shape) != 1 or len(self.ydata.shape) != 1:
            raise ValueError('Invalid shape. x and y array must be 1 dimensional.')
        if self.xdata.shape != self.ydata.shape:
            raise ValueError('Invalid shape. x and y array must have same shape.')
        self.xdata = xdata
        self.ydata = ydata

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
        self.function = self.functions[function]

    def _find_peaks(self, search_range: float = 15) -> bool:
        x_true = np.array(self.database[self.measurement][self.material])
        x_true = x_true[(x_true > self.xdata.min()) & (x_true < self.xdata.max())]  # crop
        search_ranges = [[x-search_range, x+search_range] for x in x_true]

        fitted_x = []
        found_x_true = []
        for x_ref_true, search_range in zip(x_true, search_ranges):
            # Crop
            partial = (search_range[0] < self.xdata) & (self.xdata < search_range[1])
            x_partial = self.xdata[partial]
            y_partial = self.ydata[partial]

            # Begin with finding the maximum position
            found_peaks, properties = find_peaks(y_partial, prominence=50)
            if len(found_peaks) != 1:
                print('Some peaks were not detected.')
                continue

            # Fit with Voigt based on the found peak
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

    def _train(self, dimension: int) -> None:
        self.pf.set_params(degree=dimension)
        fitted_x_poly = self.pf.fit_transform(self.fitted_x.reshape(-1, 1))

        # Train the linear model
        self.lr.fit(fitted_x_poly, np.array(self.found_x_true).reshape(-1, 1))

    def calibrate(self, dimension: int) -> bool:
        if not self._find_peaks():
            return False
        self._train(dimension)
        x = self.xdata.copy()
        x = self.pf.fit_transform(x.reshape(-1, 1))
        self.xdata = np.ravel(self.lr.predict(x))

        self.calibration_info = [self.material, dimension, self.found_x_true]
        return True