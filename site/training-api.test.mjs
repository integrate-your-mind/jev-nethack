import {readFileSync} from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {handleRequest} from './worker.mjs';
const token='fixture-ingest-token';
const digest=v=>createHash('sha256').update(v).digest('hex');
const bytes=v=>new TextEncoder().encode(v);
class FakeObject {
 constructor(value,metadata={}){this.value=value;this.size=value.byteLength;this.httpEtag=`"${digest(value)}"`;this.customMetadata=metadata;this.body=new Response(value).body;}
 async arrayBuffer(){return this.value.slice().buffer;}
 async text(){return new TextDecoder().decode(this.value);}
 writeHttpMetadata(h){h.set('Content-Length',String(this.size));}
}
class FakeBucket {
 constructor(){this.objects=new Map();this.bodyReads=0;}
 async head(key){const x=this.objects.get(key);return x?new FakeObject(x.value,x.metadata):null;}
 async get(key){this.bodyReads++;return this.head(key);}
 async put(key,value,options={}){if(this.objects.has(key))return null;this.objects.set(key,{value,metadata:options.customMetadata??{}});return this.head(key);}
 async list({prefix='',limit=100,cursor}={}){const keys=[...this.objects.keys()].filter(x=>x.startsWith(prefix)).sort();const start=Number(cursor??0);return {objects:keys.slice(start,start+limit).map(key=>({key})),truncated:start+limit<keys.length,cursor:String(start+limit)};}
}
const env=()=>({BUCKET:new FakeBucket(),INGEST_TOKEN:token});
const req=(path,method='GET',body,headers={})=>new Request(`https://example.test${path}`,{method,body,headers:{...(body?{'Content-Length':String(body.byteLength),'X-Content-SHA256':digest(body)}:{}),...headers}});
const auth={Authorization:`Bearer ${token}`};
function fixture(){
 const payload=bytes('PK\x03\x04test-framed-payload');const length=new Uint8Array(8);new DataView(length.buffer).setBigUint64(0,BigInt(payload.length));
 const hash=new Uint8Array(createHash('sha256').update(payload).digest());
 const pack=new Uint8Array([...bytes('JEVNHNPZ1\n'),...length,...hash,...payload]);
 const index=bytes(JSON.stringify({schema:'jev-nethack-transition-pack/v1',eventId:'e1',offset:10,payloadBytes:payload.length,sha256:digest(payload)})+'\n');
 const artifacts=[['transitions-000001.npzpack',pack,'application/vnd.jev-nethack.transitions+npz'],['transitions-000001.index.jsonl',index,'application/x-ndjson']];
 const manifest={schema:'jev-nethack-transition-pack/v1',schemaVersion:1,sessionId:'shard-a',broadcastId:'run-a',completed:true,startedAt:'2026-09-18T05:00:00Z',endedAt:'2026-09-18T05:01:00Z',transitionCount:1,artifacts:artifacts.map(([filename,value,contentType])=>({filename,contentType,bytes:value.length,sha256:digest(value)}))};
 return {pack,index,artifacts,manifest};
}
async function uploadArtifacts(runtime,f){for(const [name,value] of f.artifacts){const r=await handleRequest(req(`/api/training/shard-a/${name}`,'PUT',value,auth),runtime);assert.equal(r.status,201,await r.text());}}
const putManifest=(runtime,m)=>handleRequest(req('/api/training/shard-a/manifest.json','PUT',bytes(JSON.stringify(m)),auth),runtime);
test('auth, required size, checksum, and framed payload validation',async()=>{
 const runtime=env(),f=fixture(),path='/api/training/shard-a/transitions-000001.npzpack';
 assert.equal((await handleRequest(req(path,'PUT',f.pack),runtime)).status,401);
 assert.equal((await handleRequest(req(path,'PUT',f.pack,{...auth,'X-Content-SHA256':'0'.repeat(64)}),runtime)).status,422);
 const noLength=req(path,'PUT',f.pack,auth);noLength.headers.delete('Content-Length');assert.equal((await handleRequest(noLength,runtime)).status,411);
 assert.equal((await handleRequest(req(path,'PUT',f.pack,{...auth,'Content-Length':String(33*1024*1024)}),runtime)).status,413);
 const bad=f.pack.slice();bad[bad.length-1]^=1;assert.equal((await handleRequest(req(path,'PUT',bad,auth),runtime)).status,422);
 assert.equal((await handleRequest(req(path,'PUT',f.pack.slice(0,-1),auth),runtime)).status,422);
 assert.equal(runtime.BUCKET.objects.size,0);
});
test('same-body retries are idempotent and changed bodies conflict',async()=>{
 const runtime=env(),f=fixture();await uploadArtifacts(runtime,f);
 const path='/api/training/shard-a/transitions-000001.index.jsonl';
 const r=await handleRequest(req(path,'PUT',f.index,auth),runtime);assert.equal(r.status,200);assert.equal((await r.json()).alreadyStored,true);
 assert.equal((await handleRequest(req(path,'PUT',bytes('{}\n'),auth),runtime)).status,409);
});
test('manifest requires present matching objects and count/index agreement',async()=>{
 let runtime=env(),f=fixture();assert.equal((await putManifest(runtime,f.manifest)).status,409);
 await uploadArtifacts(runtime,f);assert.equal((await putManifest(runtime,{...f.manifest,transitionCount:2})).status,422);
 runtime=env();f=fixture();f.artifacts[1][1]=bytes('{}\n');f.manifest.artifacts[1].bytes=f.artifacts[1][1].length;f.manifest.artifacts[1].sha256=digest(f.artifacts[1][1]);await uploadArtifacts(runtime,f);assert.equal((await putManifest(runtime,f.manifest)).status,422);
});
test('published manifest retry, anonymous download, HEAD, and pagination',async()=>{
 const runtime=env(),f=fixture();await uploadArtifacts(runtime,f);assert.equal((await putManifest(runtime,f.manifest)).status,201);assert.equal((await putManifest(runtime,f.manifest)).status,200);
 let r=await handleRequest(req('/api/training/shard-a/transitions-000001.npzpack'),runtime);assert.equal(r.status,200);assert.equal(r.headers.get('X-Content-Type-Options'),'nosniff');assert.equal(digest(new Uint8Array(await r.arrayBuffer())),digest(f.pack));
 r=await handleRequest(req('/api/training/shard-a/transitions-000001.npzpack','HEAD'),runtime);assert.equal(r.status,200);assert.equal(await r.text(),'');assert.equal(Number(r.headers.get('Content-Length')),f.pack.length);
 runtime.BUCKET.bodyReads=0;r=await handleRequest(req('/api/training?limit=1'),runtime);const page=await r.json();assert.equal(page.manifests.length,1);assert.ok(page.cursor);assert.equal(runtime.BUCKET.bodyReads,1,'listing reads manifest only and HEADs immutable artifacts');
 r=await handleRequest(req(`/api/training?limit=1&cursor=${page.cursor}`),runtime);assert.equal((await r.json()).manifests.length,0);
});
test('listing suppresses unverified or missing artifact manifests',async()=>{
 const runtime=env(),f=fixture();await uploadArtifacts(runtime,f);await putManifest(runtime,f.manifest);
 runtime.BUCKET.objects.delete('training/shard-a/transitions-000001.index.jsonl');let r=await handleRequest(req('/api/training'),runtime);assert.equal((await r.json()).manifests.length,0);
 const unverified=env();await unverified.BUCKET.put('training/shard-a/manifest.json',bytes(JSON.stringify(f.manifest)));r=await handleRequest(req('/api/training'),unverified);assert.equal((await r.json()).manifests.length,0);
});

test('accepts the exact NPZ pack and index produced by the Python writer',async()=>{
 const runtime=env(), f=fixture(), encoded=JSON.parse(readFileSync(new URL('./fixtures/transition-pack-v1.json',import.meta.url),'utf8'));
 f.pack=new Uint8Array(Buffer.from(encoded.packBase64,'base64'));f.index=new Uint8Array(Buffer.from(encoded.indexBase64,'base64'));
 f.artifacts[0][1]=f.pack;f.artifacts[1][1]=f.index;
 for(let i=0;i<2;i++){f.manifest.artifacts[i].bytes=f.artifacts[i][1].length;f.manifest.artifacts[i].sha256=digest(f.artifacts[i][1]);}
 await uploadArtifacts(runtime,f);const response=await putManifest(runtime,f.manifest);assert.equal(response.status,201,await response.text());
});
