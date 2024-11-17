## Evaluate
# Download the Game-Theoretic Interaction Dataset and unzip in :
https://drive.google.com/drive/folders/1_8X7iMNEwCyPxwwrzvA_sD0aoYWLmUq4?usp=drive_link


```
python evaluate.py --save_dir /home/mpc/interaction_navigation_evaluate/ --eval_mode gt_mpc --sc 1
```

```
python evaluate.py --save_dir <save_directory> --eval_mode <mode: str> --sc <int>
```

# Replace <save_directory> with a local directory
# eval_mode: [gt_mpc,mpc]
# sc: [1,2,3,4,5,6,7,8]
