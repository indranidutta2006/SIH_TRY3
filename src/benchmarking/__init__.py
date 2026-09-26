"""Benchmarking and validation package for SIH26138.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Exports benchmark solvers, adapters, convergence analysis, scalability testing,
statistical stability evaluation, Pareto frontier sorting, and prediction benchmarking.
"""

from src.benchmarking.benchmark_suite import BenchmarkSuiteOrchestrator
from src.benchmarking.convergence_analysis import ConvergenceAnalyzer
from src.benchmarking.pareto_analysis import ParetoAnalyzer
from src.benchmarking.prediction_benchmark import PredictionModelBenchmarker
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.scalability_suite import ScalabilitySuite
from src.benchmarking.statistical_stability import StatisticalStabilityEvaluator
from src.benchmarking.workflow_benchmark import WorkflowBenchmarker

__all__ = [
    "BenchmarkSuiteOrchestrator",
    "QPSOBenchmarkAdapter",
    "ConvergenceAnalyzer",
    "ScalabilitySuite",
    "PredictionModelBenchmarker",
    "StatisticalStabilityEvaluator",
    "ParetoAnalyzer",
    "WorkflowBenchmarker",
]
