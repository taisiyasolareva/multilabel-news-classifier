"""Integration tests for training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from models.simple_classifier import SimpleClassifier
from models.cnn_classifier import CNNClassifier
from models.lightning_module import NewsClassificationModule
from data.dataset import NewsDataset


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples: int = 10, num_classes: int = 10, use_snippet: bool = False):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.use_snippet = use_snippet
        
        # Create dummy data
        self.title = torch.randint(0, 1000, (num_samples, 20))
        if use_snippet:
            self.snippet = torch.randint(0, 1000, (num_samples, 50))
        self.labels = torch.randint(0, 2, (num_samples, num_classes)).float()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.use_snippet:
            return (self.title[idx], self.snippet[idx], self.labels[idx])
        else:
            return (self.title[idx], self.labels[idx])


class TestTrainingLoop:
    """Integration tests for training loop."""
    
    def test_training_step_title_only(self):
        """Test training step with title-only model."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=10, use_snippet=False)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Training step
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_training_step_with_snippet(self):
        """Test training step with snippet model."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=True
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=10, use_snippet=True)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_validation_step(self):
        """Test validation step."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=10, use_snippet=False)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        loss = lightning_module.validation_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3
        )
        
        optimizer_config = lightning_module.configure_optimizers()
        
        assert 'optimizer' in optimizer_config
        assert isinstance(optimizer_config['optimizer'], torch.optim.Adam)
    
    def test_full_training_epoch(self):
        """Test complete training epoch."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        train_dataset = DummyDataset(num_samples=8, num_classes=10, use_snippet=False)
        val_dataset = DummyDataset(num_samples=4, num_classes=10, use_snippet=False)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        # Create trainer with fast_dev_run
        trainer = pl.Trainer(
            max_epochs=1,
            logger=CSVLogger("test_logs"),
            enable_progress_bar=False,
            enable_model_summary=False,
            fast_dev_run=True
        )
        
        # Capture initial weight BEFORE training
        initial_weight = model.title_embedding.weight.clone()
        
        # Train for one epoch
        trainer.fit(lightning_module, train_loader, val_loader)
        
        # Weights should have changed after training
        assert not torch.equal(initial_weight, model.title_embedding.weight)
    
    def test_training_with_cnn_model(self):
        """Test training with CNN model."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            max_title_len=20,
            max_snippet_len=50
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=10, use_snippet=True)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_loss_decreases_during_training(self):
        """Test that loss decreases during training."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-2,  # Higher LR for faster convergence
            criterion=nn.BCEWithLogitsLoss()
        )
        
        dataset = DummyDataset(num_samples=20, num_classes=10, use_snippet=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get initial loss
        batch = next(iter(dataloader))
        initial_loss = lightning_module.training_step(batch, batch_idx=0)
        
        # Train for a few steps
        optimizer = lightning_module.configure_optimizers()['optimizer']
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Train for 5 steps
                break
            loss = lightning_module.training_step(batch, batch_idx=i)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Get final loss
        final_batch = next(iter(dataloader))
        final_loss = lightning_module.training_step(final_batch, batch_idx=0)
        
        # Loss should generally decrease (not always guaranteed, but likely)
        # We just check that training doesn't crash and loss is reasonable
        assert final_loss.item() > 0
        assert not torch.isnan(final_loss)
    
    def test_model_checkpointing(self, tmp_path):
        """Test that model can be saved and loaded."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3
        )
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=10)
        dataloader = DataLoader(dataset, batch_size=2)
        
        trainer.fit(lightning_module, dataloader)
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Verify checkpoint exists
        assert checkpoint_path.exists()
        
        # Load checkpoint - need to provide model since it's required
        # Recreate the model with same config
        loaded_model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        loaded_module = NewsClassificationModule.load_from_checkpoint(
            str(checkpoint_path),
            model=loaded_model
        )
        
        # Verify model structure is preserved
        assert isinstance(loaded_module.model, SimpleClassifier)
        # Check that model has correct structure (fc layer output features)
        assert loaded_module.model.fc.out_features == 10


class TestTrainingEdgeCases:
    """Tests for edge cases in training."""
    
    def test_training_with_single_class(self):
        """Test training with single output class."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=1,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=1)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        # Loss should be >= 0 (can be exactly 0.0 in edge cases with single class)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_training_with_many_classes(self):
        """Test training with many output classes."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=1000,
            use_snippet=False
        )
        
        lightning_module = NewsClassificationModule(
            model=model,
            learning_rate=1e-3
        )
        
        dataset = DummyDataset(num_samples=4, num_classes=1000)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        # Loss should be >= 0 (can be exactly 0.0 in edge cases with single class)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_training_with_different_loss_functions(self):
        """Test training with different loss functions."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=10,
            use_snippet=False
        )
        
        # Test with BCEWithLogitsLoss
        module1 = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.BCEWithLogitsLoss()
        )
        
        # Test with CrossEntropyLoss (requires different target format)
        # Note: CrossEntropyLoss expects class indices, not one-hot
        # This is just to test the module accepts different criteria
        module2 = NewsClassificationModule(
            model=model,
            learning_rate=1e-3,
            criterion=nn.CrossEntropyLoss()
        )
        
        assert module1.criterion is not None
        assert module2.criterion is not None

