# ComfyUI-Extensions

ComfyUI 上で Irodori-TTS 系の音声生成と Kanomemo 系の画像解析を行うカスタムノード集です。

- Irodori-TTS のチェックポイントを読み込み、テキストから音声を生成
- 参照音声、VoiceDesign、CFG、Rescale、Schedule、末尾トリムなどの生成条件を追加
- Character Voice 対応チェックポイントで、キャラクター画像を条件に音声を生成
- WAV / MP3 / FLAC で音声を保存
- imgutils、WD14 ViT、MobileSAM を使った画像解析、マスク生成、キャラクター切り抜き

このリポジトリには Irodori-TTS および Character Voice のモデル本体は含まれていません。

## 動作要件

- ComfyUI（新しい Extension API を使用できるバージョン）
- Python 3.10 以降を推奨
- Irodori-TTS v4.1-Small を使う場合は PyTorch 2.10 以降
- GPU 推論を行う場合は、ComfyUI で動作する CUDA 対応 PyTorch 環境
- Git（`dacvae` と `silentcipher` を GitHub からインストールするために必要）

`torch` と `torchaudio` は ComfyUI 側の環境を利用するため、`requirements.txt` には含めていません。既存の CUDA 対応ビルドを上書きしないでください。

## インストール

PowerShell で ComfyUI のディレクトリへ移動し、次を実行します。

```powershell
Set-Location -LiteralPath "C:\path\to\ComfyUI\custom_nodes"
git clone https://github.com/ootsuka-repos/ComfyUI-Extensions.git

Set-Location -LiteralPath "C:\path\to\ComfyUI"
.\python_embeded\python.exe -m pip install -r ".\custom_nodes\ComfyUI-Extensions\requirements.txt"
```

通常の Python 仮想環境で ComfyUI を動かしている場合は、最後のコマンドを次のように置き換えます。

```powershell
python -m pip install -r ".\custom_nodes\ComfyUI-Extensions\requirements.txt"
```

インストール後に ComfyUI を再起動してください。起動ログに `ComfyUI-Extensions` の import 成功メッセージが表示されます。

## モデルの配置

Irodori-TTS または Character Voice の `.safetensors` チェックポイントを、ComfyUI の `models/checkpoints` 以下へ配置します。サブディレクトリも利用できます。

```text
ComfyUI/
└─ models/
   ├─ checkpoints/
   │  └─ irodori_tts/
   │     └─ model.safetensors
   └─ loras/
      └─ irodori_tts/
         └─ adapter.safetensors
```

`IrodoriTTS Model Loader` のモデル一覧には、ComfyUI の `checkpoints` として認識された全ファイルが表示されます。Irodori-TTS 互換チェックポイントを選択してください。

チェックポイントの `latent_dim` は読み込み時に自動判定され、対応する codec が選択されます。

| `latent_dim` | codec |
| --- | --- |
| `32` | `Aratako/Semantic-DACVAE-Japanese-32dim` |
| `128` | `facebook/dacvae-watermarked` |

tokenizer、codec、Character Voice の画像エンコーダーは初回利用時に Hugging Face から自動取得され、拡張機能内の `data` 以下へ保存されます。Kanomemo 系モデルは `ComfyUI/models/huggingface` にキャッシュされます。初回実行にはネットワーク接続が必要です。

## 最小構成

通常の音声生成は、次の2ノードだけで実行できます。

```text
IrodoriTTS Model Loader
  └─ irodori_model_config → IrodoriTTS Sampler / model_config
                                └─ audio
```

1. `IrodoriTTS Model Loader` でチェックポイントと実行デバイスを選択します。
2. `IrodoriTTS Sampler` の `text` に読み上げる文章を入力します。
3. 必要に応じて `audio` を Preview Audio または `IrodoriTTS Save Audio` へ接続します。

通常の Irodori-TTS Sampler では音声長を文章から自動推定します。旧ワークフローの `seconds` や Duration 設定が残っていても生成時には使用されません。

![最小構成](assets/base.png)

## Irodori-TTS ノード

### 生成ノード

| ノード | 役割 |
| --- | --- |
| `IrodoriTTS Model Loader` | チェックポイント、モデル/codec のデバイスと精度、runtime の保持方針を設定 |
| `IrodoriTTS Sampler` | テキストと各種条件から ComfyUI の `AUDIO` を生成 |
| `Irodori Character Voice Sampler` | Character Voice 対応チェックポイントと任意のキャラクター画像から音声を生成 |
| `IrodoriTTS Save Audio` | `AUDIO` を ComfyUI の output ディレクトリへ WAV / MP3 / FLAC 形式で保存 |

`IrodoriTTS Sampler` は同一条件を最大16件までバッチ生成できます。`decode_mode = batch` は高速になる場合がありますが、`sequential` より多くの VRAM を使います。

Character Voice Sampler では `seconds` で1～120秒の長さを指定します。画像は任意ですが、接続した場合はバッチの先頭1枚だけを条件として使用します。参照音声、VoiceDesign、LoRA、Schedule は Character Voice Sampler には接続できません。

### 条件・補助ノード

| ノード | 役割 |
| --- | --- |
| `IrodoriTTS Reference Audio` | ComfyUI の input にある音声または動画から参照音声条件を作成 |
| `IrodoriTTS VoiceDesign Config` | 対応モデルへ声質、話速、感情、話し方などの caption 条件を渡す |
| `IrodoriTTS CFG Config` | text / speaker / caption / character の CFG 強度と適用範囲を設定 |
| `IrodoriTTS Rescale Config` | truncation、rescale、speaker K/V 補正を設定 |
| `IrodoriTTS Schedule Config` | RF サンプリングの `linear` / `sway` スケジュールを設定 |
| `IrodoriTTS Trim Tail Config` | 音声末尾の無音・平坦部分を切り詰める判定値を設定 |
| `IrodoriTTS LoRA Stack` | ComfyUI の `models/loras` から Irodori-TTS 用 LoRA を選択 |
| `IrodoriTTS Emoji Picker` | Irodori-TTS で使いやすい絵文字を選ぶフロントエンド補助ノード |

現行の Irodori-TTS v4 runtime が一度に適用できる LoRA は1つです。LoRA Stack ノードを複数連結すると、Sampler はエラーにします。

参照音声に動画を指定した場合は `imageio-ffmpeg` またはシステムの `ffmpeg` で音声を抽出します。入力ファイルは ComfyUI の input ディレクトリに置いてください。

各入力値の詳細は [docs/parameters.md](docs/parameters.md) を参照してください。

## Kanomemo ノード

Kanomemo ノードは、生成処理とは独立した画像解析・後処理ノードです。モデルは初回実行時に自動ダウンロードされます。

| ノード | 入出力と用途 |
| --- | --- |
| `Kanomemo Image Analysis (imgutils)` | 画像1枚を解析し、`face` / `head` / `censor` / `nudenet` / `wd14` / `ocr` の結果を JSON で出力 |
| `Kanomemo WD14 ViT Scores` | 指定タグの WD14 ViT スコアを JSON で出力 |
| `Kanomemo Heatmap Censor (WD14 ViT)` | WD14 ViT のヒートマップを使い、対象領域を blur または pixelate |
| `Kanomemo Object Mask (MobileSAM)` | 座標で指定した矩形をプロンプトとして MobileSAM のマスクを生成 |
| `Kanomemo Character Segment (imgutils)` | ISNetIS でアニメキャラクターを抽出し、RGB 画像と前景マスクを出力 |
| `Kanomemo Save RGBA` | RGB 画像と前景マスクを合成し、透過 PNG を output ディレクトリへ保存 |

`Kanomemo Image Analysis (imgutils)`、WD14 系ノード、MobileSAM ノードは現状1枚の RGB 画像を受け取る設計です。Character Segment と Save RGBA はバッチを処理できます。Character Segment の `scale` は実装上 `1024` 固定です。

## モデル保持とメモリ設定

`IrodoriTTS Model Loader` の `runtime_cache_policy` で生成後の状態を選択できます。

| 値 | 動作 |
| --- | --- |
| `offload_after_use` | runtime を維持しつつ CPU 側へ退避。標準設定 |
| `keep_gpu` | GPU 上に維持。連続生成は速いが VRAM を占有 |
| `unload_after_use` | 生成後に runtime を破棄。再生成時は再読み込み |

低 VRAM 環境では、まず次を試してください。

- `codec_device = cpu`、`codec_precision = fp32`
- `decode_mode = sequential`
- `runtime_cache_policy = offload_after_use` または `unload_after_use`
- `batch_size = 1`
- `compile_model = false`

CPU デバイスを選んだ状態で `bf16` や `fp16` を指定した場合、Model Loader は安全のため `fp32` に変更します。

## 出力先

- 音声: `ComfyUI/output/<filename_prefix>...`
- 透過 PNG: `ComfyUI/output/<filename_prefix>...`
- Irodori tokenizer: `custom_nodes/ComfyUI-Extensions/data/tokenizers`
- Irodori codec: `custom_nodes/ComfyUI-Extensions/data/codecs/<repo_id>`
- Character Voice 画像エンコーダー: `custom_nodes/ComfyUI-Extensions/data/image_encoders`
- Kanomemo / imgutils: `ComfyUI/models/huggingface`

codec の `<repo_id>` は `/` を `_` に置き換えた名前です。

## トラブルシューティング

### ノードが表示されない

ComfyUI の起動ログで `ComfyUI-Extensions` の import エラーを確認し、ComfyUI が実際に使用している Python で依存関係を再インストールしてください。この拡張は `comfy_api.latest` を使うため、古い ComfyUI では読み込めません。

### チェックポイントが一覧にない

ファイルを `ComfyUI/models/checkpoints` 以下へ配置し、ComfyUI を再起動またはモデル一覧を更新してください。ファイルが表示されても、Irodori-TTS 互換でなければ読み込み時にエラーになります。

### 初回実行が長い

tokenizer、codec、画像エンコーダー、imgutils、MobileSAM などを初回に取得します。進捗は ComfyUI のコンソールで確認できます。ダウンロード途中で停止した場合は、ネットワーク接続と Hugging Face へのアクセスを確認してください。

### MP3 / FLAC を保存できない

ComfyUI の音声保存ヘルパーと codec 環境を利用します。まず WAV 保存を確認し、失敗する場合は ComfyUI と `torchcodec`、FFmpeg の組み合わせを確認してください。

### VRAM 不足になる

上記の低 VRAM 設定を適用してください。Kanomemo の MobileSAM は CUDA が利用可能な場合に `cuda:0` を使用するため、TTS と同じプロセスで続けて実行する場合は VRAM 使用量にも注意してください。

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) で提供されます。ダウンロードされる各モデル、データ、依存ライブラリにはそれぞれのライセンスが適用されます。
