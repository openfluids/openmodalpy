"""OpenModalPy public package."""

from openmodalpy.bsmd import BSMDAnalyzer
from openmodalpy.commands import (
    analyze_from_config,
    analyze_from_spec,
    discover_examples,
    get_example_info,
    get_method_spec,
    inspect_results,
    list_methods,
    load_case_spec,
    load_example_payload,
    run_from_config,
)
from openmodalpy.core.provenance import collect_provenance
from openmodalpy.core.results import AnalysisResults, read_results
from openmodalpy.core.threads import blas_threads, get_blas_threads, set_blas_threads
from openmodalpy.dmd import DMDAnalyzer
from openmodalpy.example_data import (
    generate_cylinder_wake,
    generate_double_gyre,
    generate_example_dataset,
    generate_taylor_green,
)
from openmodalpy.mpod import MPODAnalyzer
from openmodalpy.pod import PODAnalyzer
from openmodalpy.psd_pod import PSDPODAnalyzer
from openmodalpy.specs import (
    AnalyzeSpec,
    CaseSpec,
    DataSourceSpec,
    ExampleInfo,
    MethodInfo,
    RunCollectionSpec,
    RunOutcome,
)
from openmodalpy.spod import SPODAnalyzer
from openmodalpy.stpod import STPODAnalyzer

__version__ = "0.5.0"
__all__ = [
    "PODAnalyzer",
    "MPODAnalyzer",
    "DMDAnalyzer",
    "SPODAnalyzer",
    "BSMDAnalyzer",
    "STPODAnalyzer",
    "PSDPODAnalyzer",
    "AnalysisResults",
    "read_results",
    "collect_provenance",
    "set_blas_threads",
    "get_blas_threads",
    "blas_threads",
    "AnalyzeSpec",
    "CaseSpec",
    "DataSourceSpec",
    "ExampleInfo",
    "MethodInfo",
    "RunCollectionSpec",
    "RunOutcome",
    "analyze_from_spec",
    "analyze_from_config",
    "run_from_config",
    "discover_examples",
    "list_methods",
    "get_method_spec",
    "inspect_results",
    "load_case_spec",
    # examples
    "generate_double_gyre",
    "generate_taylor_green",
    "generate_cylinder_wake",
    "generate_example_dataset",
    "get_example_info",
    "load_example_payload",
]
