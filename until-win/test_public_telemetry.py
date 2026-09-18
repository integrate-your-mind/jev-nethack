import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from broadcast import BroadcastRecorder, write_bytes_atomic

class AtomicMetadataTests(unittest.TestCase):
    def test_failed_replace_preserves_old_manifest_and_removes_owned_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'manifest.json'
            path.write_bytes(b'{"completed":false}')
            with patch('broadcast.os.replace',side_effect=OSError('simulated crash window')):
                with self.assertRaises(OSError):write_bytes_atomic(path,b'{"completed":true}')
            self.assertEqual(b'{"completed":false}',path.read_bytes())
            self.assertEqual([path],list(Path(temporary).iterdir()))

    def test_file_is_synced_before_replace_then_parent_synced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'manifest.json';events=[]
            import os
            original=os.replace
            def replace(source,target):
                self.assertEqual(['fsync'],events)
                self.assertEqual(b'complete',source.read_bytes())
                events.append('replace');original(source,target)
            with patch('broadcast.os.fsync',side_effect=lambda fd:events.append('fsync')),patch('broadcast.os.replace',side_effect=replace):
                write_bytes_atomic(path,b'complete')
            self.assertEqual(['fsync','replace','fsync'],events)
            self.assertEqual(b'complete',path.read_bytes())

class PublicTelemetryTests(unittest.TestCase):
    def test_episode_metrics_are_typed_and_nested_evidence_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher'):
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.set_telemetry(metrics={
                    'totalActions': 5227,
                    'episodeResults': [{
                        'episodeId': 0,
                        'seed': 103,
                        'score': 138,
                        'scoreSemantics': 'official_final',
                        'officialFinalScore': 138,
                        'lastObservedScore': 137,
                        'maxObservedScore': 137,
                        'totalReward': 77.5,
                        'status': 'game_end',
                        'endStatus': 'DEATH',
                        'deathCause': 'died of starvation',
                        'deathWhile': 'fainted',
                        'terminated': True,
                        'officialScoreEvidence': {
                            'source': 'native_xlogfile',
                            'relativePath': 'nle-ttyrec/nle.4242.xlogfile',
                            'sha256': 'a' * 64,
                            'xlogBytes': 87,
                            'record': 'single_native_xlog_record',
                            'ttyrecRelativePath': 'nle-ttyrec/nle.4242.0.ttyrec3.bz2',
                            'ttyrecBytes': 123,
                            'episodeId': 0,
                            'seed': 103,
                            'rawXlog': 'PRIVATE_SENTINEL',
                        },
                        'private': 'PRIVATE_SENTINEL',
                    }]
                })
                row=rec.public_metrics['episodeResults'][0]
                self.assertEqual(138,row['officialFinalScore'])
                self.assertEqual(137,row['lastObservedScore'])
                self.assertEqual(137,row['maxObservedScore'])
                self.assertEqual(77.5,row['totalReward'])
                self.assertNotIn('PRIVATE_SENTINEL',json.dumps(row))
                self.assertNotIn('rawXlog',row['officialScoreEvidence'])
            finally: rec.close()

    def test_malformed_public_metric_types_and_unsafe_evidence_are_omitted(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher'):
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.set_telemetry(metrics={'episodeResults': [{
                    'episodeId': True,
                    'seed': -1,
                    'score': '138',
                    'totalReward': float('nan'),
                    'status': 'arbitrary-private-status',
                    'endStatus': 'death\nPRIVATE_SENTINEL',
                    'deathCause': 'x' * 300,
                    'officialScoreError': 'arbitrary-private-error',
                    'officialScoreEvidence': {
                        'source': 'native_xlogfile',
                        'relativePath': '../../private',
                        'sha256': 'a' * 64,
                        'xlogBytes': 87,
                        'record': 'single_native_xlog_record',
                        'ttyrecRelativePath': 'nle-ttyrec/nle.1.0.ttyrec3.bz2',
                        'ttyrecBytes': 1,
                        'episodeId': 0,
                        'seed': 103,
                    },
                }]})
                self.assertEqual({},rec.public_metrics['episodeResults'][0])
                self.assertNotIn('PRIVATE_SENTINEL',json.dumps(rec.public_metrics))
            finally: rec.close()

    def test_score_evidence_must_match_row_identity_and_exact_bound_paths(self):
        valid_evidence = {
            'source': 'native_xlogfile',
            'relativePath': 'nle-ttyrec/nle.4242.xlogfile',
            'sha256': 'a' * 64,
            'xlogBytes': 87,
            'record': 'single_native_xlog_record',
            'ttyrecRelativePath': 'nle-ttyrec/nle.4242.0.ttyrec3.bz2',
            'ttyrecBytes': 123,
            'episodeId': 7,
            'seed': 103,
        }
        invalid_evidence = {
            'foreign episode id': {**valid_evidence, 'episodeId': 8},
            'foreign seed': {**valid_evidence, 'seed': 104},
            'arbitrary safe xlog path': {
                **valid_evidence,
                'relativePath': 'nle-ttyrec/other.xlogfile',
            },
            'arbitrary safe ttyrec path': {
                **valid_evidence,
                'ttyrecRelativePath': 'nle-ttyrec/other.ttyrec3.bz2',
            },
            'ttyrec pid mismatch': {
                **valid_evidence,
                'ttyrecRelativePath': 'nle-ttyrec/nle.9999.0.ttyrec3.bz2',
            },
        }
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher'):
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                for label, evidence in invalid_evidence.items():
                    with self.subTest(label=label):
                        rec.set_telemetry(metrics={'episodeResults': [{
                            'episodeId': 7,
                            'seed': 103,
                            'officialScoreEvidence': evidence,
                        }]})
                        row=rec.public_metrics['episodeResults'][0]
                        self.assertEqual(7,row['episodeId'])
                        self.assertEqual(103,row['seed'])
                        self.assertNotIn('officialScoreEvidence',row)
            finally: rec.close()

    def test_huge_integer_reward_is_omitted_without_interrupting_telemetry(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher'):
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.set_telemetry(metrics={'episodeResults': [{
                    'episodeId': 7,
                    'seed': 103,
                    'totalReward': 10 ** 10000,
                    'status': 'game_end',
                }]})
                row=rec.public_metrics['episodeResults'][0]
                self.assertEqual(7,row['episodeId'])
                self.assertEqual('game_end',row['status'])
                self.assertNotIn('totalReward',row)
            finally: rec.close()

    def test_probabilities_keep_ids_labels_and_exclude_private_provenance(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher') as publisher:
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.decision(episode=2,step=12,decision={'choice':'a0','confidence':.8,'model':'jev-1.13.0','probabilities':{'a0':.7,'a1':.29},'request_body_base64':'PRIVATE_SENTINEL'},criteria={'a0':'Move north','a1':'Select inventory item'})
                rec.set_telemetry(metrics={'totalActions':13,'completedEpisodes':2,'ascensions':0,'secret':'PRIVATE_SENTINEL','episodeResults':[{'episodeId':1,'score':20,'status':'game_end','private':'PRIVATE_SENTINEL'}]},runtime_status={'phase':'playing'})
                rec.observed_frame(state={'terminal':'@','player':{'score':4}},phase='after_action',episode=2,step=12,action={'index':0})
                public=publisher.return_value.submit.call_args.args[0]
                self.assertEqual({'a0':.7,'a1':.29},public['decision']['probabilities'])
                self.assertEqual('Move north',public['decision']['criteria']['a0'])
                self.assertEqual(12,public['decision']['step'])
                self.assertEqual(2,public['metrics']['completedEpisodes'])
                self.assertNotIn('PRIVATE_SENTINEL',json.dumps(public))
            finally: rec.close()

    def test_retry_status_does_not_falsify_frame_capture_time(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher') as publisher:
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.observed_frame(state={'terminal':'@','player':{'score':4}},phase='before_action',episode=2,step=12)
                initial=publisher.return_value.submit.call_args.args[0]
                rec.report_status('retrying',error_type='JevTransportError',retry_at='2026-09-18T06:30:00Z')
                status=publisher.return_value.submit.call_args.args[0]
                self.assertEqual(initial['capturedAt'],status['capturedAt'])
                self.assertGreater(status['sequence'],initial['sequence'])
                self.assertEqual('retrying',status['recovery']['phase'])
                self.assertEqual(initial['step'],status['step'])
            finally: rec.close()

    def test_invalid_distribution_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary, patch('broadcast.LivePublisher'):
            rec=BroadcastRecorder(Path(temporary)/'run',stream_id='run-a',ingest=object())
            try:
                rec.decision(episode=0,step=0,decision={'choice':'a0','probabilities':{'a0':float('nan')}})
                self.assertIsNone(rec.last_decision)
                rec.decision(episode=0,step=1,decision={'choice':'a0','probabilities':{'a0':1}})
                self.assertIsNotNone(rec.last_decision)
                rec.decision(episode=0,step=2,decision={'choice':'a0'})
                self.assertIsNone(rec.last_decision)
            finally: rec.close()
if __name__=='__main__':unittest.main()
