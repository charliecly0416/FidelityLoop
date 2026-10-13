import copy
import math
import unittest
from verify_adapter import adapt, evaluate

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.inputs=[dict(id=str(i),arrived_at=t,num_prefill_tokens=128,num_decode_tokens=32)
                     for i,t in enumerate([0.0,1.0,2.0])]
        self.outputs=[dict(r,scheduled_at=r["arrived_at"],completed=True,
                           completed_at=t,num_processed_tokens=160)
                      for r,t in zip(self.inputs,[0.5,2.0,3.0])]
    def test_inclusive_deadline_and_cutoff(self):
        result,rows,records=evaluate(self.inputs,self.outputs,
            dict(horizon=2.0,relative_deadline_seconds=0.5))
        self.assertEqual(result["independent_tally"],dict(arrivals=2,completed=2,on_time=1,late=1,unfinished=0))
        self.assertEqual(result["mapping"]["excluded_at_or_after_cutoff"],1)
        self.assertFalse(result["P1plus_applicable"])
    def test_future_completion_redacted_not_late(self):
        result,rows,records=evaluate(self.inputs,self.outputs,
            dict(horizon=1.5,relative_deadline_seconds=0.1))
        self.assertIsNone(records["1"]["completed_at"])
        self.assertEqual(result["independent_tally"]["unfinished"],1)
        self.assertEqual(result["independent_tally"]["late"],1)
    def test_missing_and_extra_rows_rejected(self):
        for outputs in [self.outputs[:-1],self.outputs+[dict(self.outputs[0],id="extra")]]:
            with self.assertRaises(ValueError):adapt(self.inputs,outputs,horizon=60,relative_deadline_seconds=60)
    def test_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):adapt(self.inputs,self.outputs+[self.outputs[0]],horizon=60,relative_deadline_seconds=60)
    def test_unfinished_zero_is_not_completion(self):
        out=copy.deepcopy(self.outputs);out[0].update(completed=False,completed_at=0,num_processed_tokens=10)
        rows,records,_=adapt(self.inputs,out,horizon=60,relative_deadline_seconds=60)
        self.assertIsNone(records["0"]["completed_at"])
    def test_identity_payload_and_time_corruption_rejected(self):
        corruptions=[("arrived_at",0.01),("num_decode_tokens",31),("completed_at",-1),
                     ("completed_at",float("nan")),("scheduled_at",1),("completed",1),
                     ("num_processed_tokens",159)]
        for key,value in corruptions:
            with self.subTest(key=key,value=value):
                outputs=copy.deepcopy(self.outputs);outputs[0][key]=value
                with self.assertRaises(ValueError):adapt(self.inputs,outputs,horizon=60,relative_deadline_seconds=60)
    def test_order_does_not_define_identity(self):
        a=evaluate(self.inputs,self.outputs,dict(horizon=60,relative_deadline_seconds=60))[0]
        b=evaluate(self.inputs,list(reversed(self.outputs)),dict(horizon=60,relative_deadline_seconds=60))[0]
        self.assertEqual(a,b)

    def test_native_integral_float_tokens_are_normalized(self):
        inputs=copy.deepcopy(self.inputs);outputs=copy.deepcopy(self.outputs)
        for row in inputs+outputs:
            row.update(num_prefill_tokens=128.0,num_decode_tokens=32.0)
        rows,_,_=adapt(inputs,outputs,horizon=60,relative_deadline_seconds=60)
        self.assertIs(type(rows[0]["input_tokens"]),int)
        self.assertEqual(rows[0]["input_tokens"],128)

    def test_fractional_nonfinite_and_boolean_tokens_rejected(self):
        for value in [128.5,float("nan"),float("inf"),True,0,-1]:
            for side in ["input","output"]:
                with self.subTest(value=value,side=side):
                    inputs=copy.deepcopy(self.inputs);outputs=copy.deepcopy(self.outputs)
                    (inputs if side=="input" else outputs)[0]["num_prefill_tokens"]=value
                    with self.assertRaises(ValueError):
                        adapt(inputs,outputs,horizon=60,relative_deadline_seconds=60)

if __name__=="__main__":unittest.main()
