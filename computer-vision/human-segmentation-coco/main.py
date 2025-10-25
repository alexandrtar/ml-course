import argparse
import yaml
import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Human Instance Segmentation with YOLO')
    parser.add_argument('--config', type=str, default='config/default.yaml', 
                       help='Path to config file')
    parser.add_argument('--num_images', type=int, default=5, 
                       help='Number of images to process')
    parser.add_argument('--model_type', type=str, default='yolo', 
                       choices=['yolo', 'mask_rcnn'], help='Model type')
    parser.add_argument('--conf_threshold', type=float, default=0.25, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Could not load config: {e}, using defaults")
        config = {}
    
    # Override with command line args
    if args.num_images: config['num_images'] = args.num_images
    if args.model_type: config['model_type'] = args.model_type
    if args.conf_threshold: config['conf_threshold'] = args.conf_threshold
    
    print("🚀 Starting Human Instance Segmentation Pipeline...")
    print(f"📁 Config: {config}")
    
    try:
        # Initialize components
        from data_loader import COCODataLoader
        from segmentation import HumanSegmentator
        from evaluation import SegmentationEvaluator
        from visualization import ResultsVisualizer
        
        data_loader = COCODataLoader()
        segmentator = HumanSegmentator(
            model_type=config.get('model_type', 'yolo'),
            conf_threshold=config.get('conf_threshold', 0.25)
        )
        evaluator = SegmentationEvaluator()
        visualizer = ResultsVisualizer()
        
        # Download and initialize data
        print("📥 Downloading dataset...")
        data_loader.download_dataset()
        
        print("🔧 Initializing COCO API...")
        if not data_loader.initialize_coco():
            print("❌ Failed to initialize COCO API")
            return
        
        # Get human images
        img_ids = data_loader.get_human_images(config.get('num_images', 5))
        
        if not img_ids:
            print("❌ No human images found")
            return
            
        print(f"🎯 Processing {len(img_ids)} images with humans...")
        
        # Process each image
        all_metrics = []
        processed_count = 0
        
        for i, img_id in enumerate(img_ids):
            print(f"\n📸 Processing image {i+1}/{len(img_ids)} (ID: {img_id})")
            
            try:
                # Load data
                data = data_loader.load_image_and_mask(img_id)
                
                if data['image'] is None or data['mask'] is None:
                    print(f"⚠️  Skipping image {img_id} - could not load")
                    continue
                
                # Perform segmentation
                print("   🤖 Running segmentation...")
                pred_mask = segmentator.segment(data['image'])
                
                # Evaluate
                metrics = evaluator.evaluate_single(data['mask'], pred_mask)
                all_metrics.append(metrics)
                
                # Visualize and save
                save_path = visualizer.save_results(
                    data['image'], data['mask'], pred_mask, metrics, img_id,
                    subfolder=config.get('model_type', 'yolo')
                )
                
                # Print progress
                print(f"   ✅ Processed - IoU: {metrics['iou']:.3f}")
                print(f"   💾 Saved to: {save_path}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing image {img_id}: {e}")
                continue
        
        # Final summary
        if processed_count > 0:
            print(f"\n📊 FINAL RESULTS (Processed {processed_count} images):")
            print("=" * 50)
            aggregated = evaluator.aggregate_metrics(all_metrics)
            evaluator.print_metrics(aggregated)
            
            # Save overall results
            results_file = f"results/{config.get('model_type', 'yolo')}_summary.txt"
            with open(results_file, 'w') as f:
                f.write("Human Instance Segmentation Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Processed images: {processed_count}\n")
                f.write(f"Model: {config.get('model_type', 'yolo')}\n")
                f.write(f"Confidence threshold: {config.get('conf_threshold', 0.25)}\n\n")
                for key, value in aggregated.items():
                    f.write(f"{key}: {value:.4f}\n")
            
            print(f"📁 Detailed results saved to: {results_file}")
        else:
            print("❌ No images were successfully processed")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()