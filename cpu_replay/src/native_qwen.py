"""Native Qwen factory, using the accepted service-screen RF and operator limits."""
import atexit
import csv
import math
from pathlib import Path

from service_predictor import FROZEN_CONFIG, verify_profile
import qwen_registration
from vidur.config import (ClusterConfig, MetricsConfig, ReplicaConfig, SarathiSchedulerConfig,
    SimulationConfig, TraceRequestGeneratorConfig, RandomForrestExecutionTimePredictorConfig)
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.execution_time_predictor.random_forrest_execution_time_predictor import RandomForrestExecutionTimePredictor
from vidur.simulator import Simulator
from vidur.entities import Request, Replica
from vidur.events import BaseEvent
from vidur.types import ExecutionTimePredictorType
from vidur.utils.random import set_seeds

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'inputs/profile'
PREDICTORS = []


def check_domain(query):
    if (query['compute_tokens'] > 8192 or query['kv_save_tokens'] > 8192
            or query['aggregate_prefill_tokens'] > 4096 or query['aggregate_prefill_kv'] != 0
            or query['decode_batch'] > 4 or query['decode_kv'] > 2304):
        raise ValueError(f'Runtime query outside frozen measured domain: {query}')


class DomainBoundedPredictor(RandomForrestExecutionTimePredictor):
    def _predict_for_compute_models(self):
        original = self._max_tokens
        self._max_tokens = FROZEN_CONFIG['compute_lookup_max_tokens']
        try:
            return super()._predict_for_compute_models()
        finally:
            self._max_tokens = original

    def get_execution_time(self, batch, pipeline_stage):
        prefill = self._get_batch_prefill_attention_params(batch)
        decode_batch, decode_kv = self._get_batch_decode_attention_params(batch)
        query = dict(compute_tokens=batch._total_num_tokens_rounded,
                     kv_save_tokens=sum(batch.num_tokens),
                     aggregate_prefill_tokens=round(math.sqrt(sum(chunk * chunk for _, chunk in prefill))),
                     aggregate_prefill_kv=sum(kv for kv, _ in prefill),
                     decode_batch=decode_batch, decode_kv=decode_kv)
        check_domain(query)
        audit = self.audit
        audit['native_batch_count'] += 1
        audit['mixed_prefill_decode_batches'] += bool(prefill and decode_batch)
        for key, value in query.items():
            audit['maximum_observed_query'][key] = max(value, audit['maximum_observed_query'].get(key, 0))
        return super().get_execution_time(batch, pipeline_stage)


def make_simulator(rows, out):
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    verify_profile(PROFILE, ROOT / 'inputs/service_protocol.json')
    trace = out / 'native_input.csv'
    with trace.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arrived_at', 'num_prefill_tokens', 'num_decode_tokens'])
        writer.writerows((r['arrival_s'], r['input_tokens'], r['max_output_tokens']) for r in rows)
    Request._id = Replica._id = -1
    BaseEvent._id = 0
    audit = dict(native_batch_count=0, mixed_prefill_decode_batches=0, maximum_observed_query={})

    def factory(**kwargs):
        if not PREDICTORS:
            PREDICTORS.append(DomainBoundedPredictor(**kwargs))
        PREDICTORS[0].audit = audit
        return PREDICTORS[0]

    config = SimulationConfig(seed=42, cluster_config=ClusterConfig(num_replicas=2,
        replica_config=ReplicaConfig(model_name=qwen_registration.Qwen25_7BInstructConfig.get_name(),
            device='a100', tensor_parallel_size=1, num_pipeline_stages=1),
        replica_scheduler_config=SarathiSchedulerConfig(batch_size_cap=4, chunk_size=16384, block_size=16)),
        request_generator_config=TraceRequestGeneratorConfig(trace_file=str(trace), max_tokens=2560),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            compute_input_file=str(PROFILE / 'mlp.csv'), attention_input_file=str(PROFILE / 'attention.csv'),
            no_cache=True, num_estimators=[100], max_depth=[8], min_samples_split=[2], k_fold_cv_splits=3,
            num_training_job_threads=2, prediction_max_prefill_chunk_size=4096,
            prediction_max_batch_size=4, prediction_max_tokens_per_request=2560,
            kv_cache_prediction_granularity=64, skip_cpu_overhead_modeling=True,
            attention_prefill_batching_overhead_fraction=0.1, attention_decode_batching_overhead_fraction=0.1),
        metrics_config=MetricsConfig(output_dir=str(out/'native'), cache_dir=str(ROOT/'runtime/model_cache'),
            write_metrics=False, enable_chrome_trace=False, write_json_trace=False, store_plots=False))
    key = ExecutionTimePredictorType.RANDOM_FORREST
    original = ExecutionTimePredictorRegistry.get_class(key)
    ExecutionTimePredictorRegistry.unregister(key)
    ExecutionTimePredictorRegistry.register(key, factory)
    try:
        set_seeds(42)
        sim = Simulator(config)
        atexit.unregister(sim._write_output)
    finally:
        ExecutionTimePredictorRegistry.unregister(key)
        ExecutionTimePredictorRegistry.register(key, original)
    requests = sorted((e._request for _, e in sim._event_queue), key=lambda r: r.id)
    assert len(requests) == len(rows)
    for spec, req in zip(rows, requests):
        assert req.size == (spec['input_tokens'], spec['max_output_tokens'])
        assert math.isclose(req.arrived_at, spec['arrival_s'], rel_tol=0, abs_tol=1e-12)
    sim.domain_audit = audit
    return sim, {row['request_id']: req for row, req in zip(rows, requests)}
