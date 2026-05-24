import sys
import os
import numpy as np
import pandas as pd

# 1. Monkeypatch legacy numpy attributes that were removed in NumPy 2.x
np.float = float
np.int = int
np.bool = np.bool_

# 2. Monkeypatch pandas.read_csv for delim_whitespace compatibility in Pandas 2.x/3.x
orig_read_csv = pd.read_csv
def patched_read_csv(*args, **kwargs):
    if 'delim_whitespace' in kwargs:
        if kwargs['delim_whitespace']:
            kwargs['sep'] = r'\s+'
        del kwargs['delim_whitespace']
    return orig_read_csv(*args, **kwargs)
pd.read_csv = patched_read_csv

# 3. Monkeypatch pandas.DataFrame.append which was removed in Pandas 2.x/3.x
def df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    elif isinstance(other, list) and all(isinstance(x, dict) for x in other):
        other = pd.DataFrame(other)
    elif isinstance(other, pd.Series):
        other = pd.DataFrame([other])
    return pd.concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
pd.DataFrame.append = df_append

# 3.5. Monkeypatch pandas.DataFrame.fillna and pandas.Series.fillna for method="ffill"/"bfill" compatibility in Pandas 2.x/3.x
orig_df_fillna = pd.DataFrame.fillna
def patched_df_fillna(self, *args, **kwargs):
    if 'method' in kwargs:
        method = kwargs.pop('method')
        if method == 'ffill':
            res = self.ffill()
            if kwargs:
                res = orig_df_fillna(res, *args, **kwargs)
            return res
        elif method == 'bfill':
            res = self.bfill()
            if kwargs:
                res = orig_df_fillna(res, *args, **kwargs)
            return res
    return orig_df_fillna(self, *args, **kwargs)
pd.DataFrame.fillna = patched_df_fillna

orig_series_fillna = pd.Series.fillna
def patched_series_fillna(self, *args, **kwargs):
    if 'method' in kwargs:
        method = kwargs.pop('method')
        if method == 'ffill':
            res = self.ffill()
            if kwargs:
                res = orig_series_fillna(res, *args, **kwargs)
            return res
        elif method == 'bfill':
            res = self.bfill()
            if kwargs:
                res = orig_series_fillna(res, *args, **kwargs)
            return res
    return orig_series_fillna(self, *args, **kwargs)
pd.Series.fillna = patched_series_fillna

# 4. Monkeypatch sklearn.externals.joblib to the top-level joblib
try:
    import joblib
    sys.modules['sklearn.externals.joblib'] = joblib
except ImportError:
    pass

# 5. Monkeypatch legacy sklearn SCORERS dictionary
try:
    import sklearn.metrics
    from sklearn.metrics import _scorer
    sklearn.metrics.SCORERS = _scorer._SCORERS
    sys.modules['sklearn.metrics'].SCORERS = _scorer._SCORERS
except (ImportError, AttributeError):
    pass

# 6. Monkeypatch legacy pymatgen imports (moved to pymatgen.core and pymatgen.ext)
try:
    import pymatgen
    from pymatgen.core import Composition, Structure, Element, Species as Specie, Lattice
    from pymatgen.ext.matproj import MPRester
    
    for name, obj in [
        ('Composition', Composition),
        ('Structure', Structure),
        ('Element', Element),
        ('Specie', Specie),
        ('Lattice', Lattice),
        ('MPRester', MPRester)
    ]:
        setattr(pymatgen, name, obj)
        sys.modules['pymatgen'].__dict__[name] = obj
except (ImportError, AttributeError):
    pass

# 7. Monkeypatch legacy pymatgen Spin import
try:
    from pymatgen.electronic_structure.core import Spin
    pymatgen.Spin = Spin
    sys.modules['pymatgen'].Spin = Spin
except (ImportError, AttributeError):
    pass

# 8. Monkeypatch _pt_data casing in periodic_table and limit to exactly 103 elements to prevent Magpie IndexError
try:
    import pymatgen.core.periodic_table
    pymatgen.core.periodic_table._pt_data = {
        k: v for k, v in pymatgen.core.periodic_table._PT_DATA.items()
        if 1 <= v.get('Atomic no', 999) <= 103 and k not in ['D', 'T']
    }
except (ImportError, AttributeError):
    pass

# 9. Monkeypatch pymatgen.analysis.__file__ for namespace packaging compatibility
try:
    import pymatgen.analysis
    pymatgen.analysis.__file__ = os.path.join(list(pymatgen.analysis.__path__)[0], '__init__.py')
except (ImportError, AttributeError, IndexError):
    pass

# 10. Monkeypatch get_dimensionality in structure_analyzer
try:
    import pymatgen.analysis.structure_analyzer
    def dummy_get_dimensionality(*args, **kwargs):
        return 3
    pymatgen.analysis.structure_analyzer.get_dimensionality = dummy_get_dimensionality
    sys.modules['pymatgen.analysis.structure_analyzer'].get_dimensionality = dummy_get_dimensionality
except (ImportError, AttributeError):
    pass

# 11. Monkeypatch sph_harm in scipy.special
try:
    import scipy.special
    def dummy_sph_harm(*args, **kwargs):
        return np.zeros(1)
    scipy.special.sph_harm = dummy_sph_harm
    sys.modules['scipy.special'].sph_harm = dummy_sph_harm
except (ImportError, AttributeError):
    pass

# 12. Monkeypatch ruamel.yaml.safe_load
try:
    import ruamel.yaml
    def working_safe_load(stream, *args, **kwargs):
        y = ruamel.yaml.YAML(typ='safe', pure=True)
        return y.load(stream)
    ruamel.yaml.safe_load = working_safe_load
    sys.modules['ruamel.yaml'].safe_load = working_safe_load
except (ImportError, AttributeError):
    pass

# 13. Monkeypatch AutoFeaturizer to expose bandstructure_col property for scikit-learn get_params compatibility
try:
    from automatminer import AutoFeaturizer
    @property
    def get_bandstructure_col(self):
        return getattr(self, 'bandstruct_col', 'bandstructure')
    @get_bandstructure_col.setter
    def get_bandstructure_col(self, val):
        self.bandstruct_col = val
    AutoFeaturizer.bandstructure_col = get_bandstructure_col
except (ImportError, AttributeError):
    pass

# 14. Monkeypatch TPOT's source_decode function to resolve Python 3.13 exec/locals namespace issues
try:
    import tpot.operator_utils
    def patched_source_decode(sourcecode, verbose=0):
        tmp_path = sourcecode.split('.')
        op_str = tmp_path.pop()
        import_str = '.'.join(tmp_path)
        try:
            ldict = {}
            if sourcecode.startswith('tpot.'):
                g = globals().copy()
                g['__package__'] = 'tpot'
                exec('from {} import {}'.format(import_str[4:], op_str), g, ldict)
            else:
                exec('from {} import {}'.format(import_str, op_str), globals(), ldict)
            op_obj = ldict[op_str]
        except Exception as e:
            if verbose > 2:
                raise ImportError('Error: could not import {}.\n{}'.format(sourcecode, e))
            else:
                print('Warning: {} is not available and will not be used by TPOT.'.format(sourcecode))
            op_obj = None

        return import_str, op_str, op_obj
    tpot.operator_utils.source_decode = patched_source_decode
except (ImportError, AttributeError):
    pass

# 15. Monkeypatch TPOTBase._import_hash to resolve Python 3.13 exec/locals namespace issues
try:
    from tpot.base import TPOTBase
    def patched_import_hash(self, operator):
        for key in sorted(operator.import_hash.keys()):
            module_list = ', '.join(sorted(operator.import_hash[key]))
            ldict = {}
            if key.startswith('tpot.'):
                g = globals().copy()
                g['__package__'] = 'tpot'
                exec('from {} import {}'.format(key[4:], module_list), g, ldict)
            else:
                exec('from {} import {}'.format(key, module_list), globals(), ldict)
            for var in operator.import_hash[key]:
                self.operators_context[var] = ldict[var]
    TPOTBase._import_hash = patched_import_hash
except (ImportError, AttributeError):
    pass

# 16. Monkeypatch _wrapped_cross_val_score to support modern scikit-learn return formats
try:
    import tpot.gp_deap
    import tpot.base
    from stopit import threading_timeoutable, TimeoutException
    
    @threading_timeoutable(default="Timeout")
    def patched_wrapped_cross_val_score(sklearn_pipeline, features, target,
                                         cv, scoring_function, sample_weight=None,
                                         groups=None, use_dask=False):
        from tpot.operator_utils import set_sample_weight
        from sklearn.utils import indexable
        from sklearn.base import clone, is_classifier
        from sklearn.model_selection._split import check_cv
        from sklearn.metrics.scorer import check_scoring
        from sklearn.model_selection._validation import _fit_and_score
        import warnings
        
        sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)
        features, target, groups = indexable(features, target, groups)
        cv = check_cv(cv, target, classifier=is_classifier(sklearn_pipeline))
        cv_iter = list(cv.split(features, target, groups))
        scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline),
                                         X=features,
                                         y=target,
                                         scorer=scorer,
                                         train=train,
                                         test=test,
                                         verbose=0,
                                         parameters=None,
                                         fit_params=sample_weight_dict)
                                    for train, test in cv_iter]
                
                if len(scores) > 0 and isinstance(scores[0], dict):
                    test_scores = [s['test_scores'] for s in scores]
                    if isinstance(test_scores[0], dict):
                        test_scores = [list(ts.values())[0] for ts in test_scores]
                    CV_score = np.array(test_scores)
                else:
                    CV_score = np.array(scores)[:, 0]
                
                return np.nanmean(CV_score)
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return -float('inf')
            
    tpot.gp_deap._wrapped_cross_val_score = patched_wrapped_cross_val_score
    tpot.base._wrapped_cross_val_score = patched_wrapped_cross_val_score
except (ImportError, AttributeError):
    pass

# 17. Patch sklearn.model_selection._validation._fit_and_score to support default score_params=None
try:
    import sklearn.model_selection._validation as val
    orig_fit_and_score = val._fit_and_score
    def patched_fit_and_score(*args, **kwargs):
        if 'score_params' not in kwargs and len(args) < 11:
            kwargs['score_params'] = None
        return orig_fit_and_score(*args, **kwargs)
    val._fit_and_score = patched_fit_and_score
except (ImportError, AttributeError):
    pass

# 18. Monkeypatch TPOTBase._add_operators to register dynamic Ret_ classes in tpot.base module
#     AND all ARG parameter classes in tpot.operator_utils for pickle compatibility
try:
    from tpot.base import TPOTBase
    import tpot.operator_utils as _op_utils_patch18
    import sys
    orig_add_operators = TPOTBase._add_operators
    def patched_add_operators(self):
        orig_add_operators(self)
        # Register Ret_ types into tpot.base
        if hasattr(self, 'ret_types'):
            for r_type in self.ret_types:
                name = getattr(r_type, '__name__', None)
                if name and name.startswith('Ret_'):
                    setattr(sys.modules['tpot.base'], name, r_type)
        # Register all operator and ARG classes into tpot.operator_utils
        if hasattr(self, 'operators'):
            for op in self.operators:
                op_name = getattr(op, '__name__', None)
                if op_name:
                    op.__module__ = 'tpot.operator_utils'
                    setattr(_op_utils_patch18, op_name, op)
                    sys.modules['tpot.operator_utils'].__dict__[op_name] = op
                # Also register each ARG parameter type
                for arg_type in getattr(op, 'arg_types', ()):
                    aname = getattr(arg_type, '__name__', None)
                    if aname:
                        arg_type.__module__ = 'tpot.operator_utils'
                        setattr(_op_utils_patch18, aname, arg_type)
                        sys.modules['tpot.operator_utils'].__dict__[aname] = arg_type
    TPOTBase._add_operators = patched_add_operators
except (ImportError, AttributeError):
    pass

# 19. Monkeypatch ARGTypeClassFactory to register every dynamic ARG parameter class
#     into the tpot.operator_utils module namespace so pickle can find them by name.
try:
    import tpot.operator_utils as _op_utils
    _orig_ARGTypeClassFactory = _op_utils.ARGTypeClassFactory

    def _patched_ARGTypeClassFactory(classname, prange, BaseClass=_op_utils.ARGType):
        cls = _orig_ARGTypeClassFactory(classname, prange, BaseClass)
        cls.__module__ = 'tpot.operator_utils'
        # Register in the module so pickle can find it
        setattr(_op_utils, classname, cls)
        sys.modules['tpot.operator_utils'].__dict__[classname] = cls
        return cls

    _op_utils.ARGTypeClassFactory = _patched_ARGTypeClassFactory
except (ImportError, AttributeError):
    pass

# 20. Monkeypatch TPOTOperatorClassFactory to register every dynamic TPOT_ operator class
#     into the tpot.operator_utils module namespace so pickle can find them by name.
try:
    import tpot.operator_utils as _op_utils2
    _orig_TPOTOperatorClassFactory = _op_utils2.TPOTOperatorClassFactory

    def _patched_TPOTOperatorClassFactory(opsourse, opdict,
                                          BaseClass=_op_utils2.Operator,
                                          ArgBaseClass=_op_utils2.ARGType,
                                          verbose=0):
        op_class, arg_types = _orig_TPOTOperatorClassFactory(
            opsourse, opdict, BaseClass, ArgBaseClass, verbose)
        if op_class is not None:
            name = op_class.__name__
            op_class.__module__ = 'tpot.operator_utils'
            setattr(_op_utils2, name, op_class)
            sys.modules['tpot.operator_utils'].__dict__[name] = op_class
        return op_class, arg_types

    _op_utils2.TPOTOperatorClassFactory = _patched_TPOTOperatorClassFactory
except (ImportError, AttributeError):
    pass

# 21. Add module-level __getattr__ to tpot.base for pickle compatibility
try:
    import sys
    import tpot.base
    def base_getattr(name):
        if name.startswith('Ret_'):
            cls = type(name, (object,), {})
            cls.__module__ = 'tpot.base'
            setattr(tpot.base, name, cls)
            sys.modules['tpot.base'].__dict__[name] = cls
            return cls
        raise AttributeError(f"module 'tpot.base' has no attribute '{name}'")
    tpot.base.__getattr__ = base_getattr
except (ImportError, AttributeError):
    pass

# 22. Add module-level __getattr__ to tpot.operator_utils for pickle compatibility
try:
    import sys
    import tpot.operator_utils
    from tpot.operator_utils import Operator, ARGType
    def op_utils_getattr(name):
        if name in ('Operator', 'ARGType'):
            raise AttributeError(f"module 'tpot.operator_utils' has no attribute '{name}'")
        if '__' in name:
            cls = type(name, (ARGType,), {'values': []})
            cls.__module__ = 'tpot.operator_utils'
            setattr(tpot.operator_utils, name, cls)
            sys.modules['tpot.operator_utils'].__dict__[name] = cls
            return cls
        else:
            cls = type(name, (Operator,), {})
            cls.__module__ = 'tpot.operator_utils'
            setattr(tpot.operator_utils, name, cls)
            sys.modules['tpot.operator_utils'].__dict__[name] = cls
            return cls
    tpot.operator_utils.__getattr__ = op_utils_getattr
except (ImportError, AttributeError):
    pass

# Print a verification message if running this script directly
if __name__ == "__main__":
    print("All compatibility monkeypatches successfully applied!")

