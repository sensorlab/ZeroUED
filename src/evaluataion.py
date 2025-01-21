from abc import ABC, abstractmethod

class Trainer(ABC):
    """
    Abstract base class for training models.

    This class defines the interface that all trainer classes should implement.
    """

    @abstractmethod
    def train_epoch(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to train on (CPU or GPU).
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.

        Returns:
            Dict: Training metrics for the epoch, outputs, and targets.
        """
        pass

    @abstractmethod
    def validate(self, model, val_loader, criterion, device):
        """
        Validate the model.

        Args:
            model (torch.nn.Module): The model to validate.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to validate on (CPU or GPU).

        Returns:
            Dict: Validation metrics, outputs, and targets.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model, optimizer, epoch, file_path):
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): The current epoch.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_checkpoint(self, model, optimizer, file_path):
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): The model to load.
            optimizer (torch.optim.Optimizer): The optimizer.
            file_path (str): Path to the checkpoint file.

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        pass