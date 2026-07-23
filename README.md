Comando per effettuare il training: prog/agente.py --pretrained-backbone prog/pretrained/backbone_pretrained_backbone.pt --n-epochs 400 --batch-size 64 --tau-iou-start 0.6

Comando per effettuare il test: prog/agente.py --pretrained-backbone prog/pretrained/backbone_pretrained_backbone.pt --test --model \<checkpoint>