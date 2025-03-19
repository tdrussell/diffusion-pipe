from torch import nn

from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import LayerSpec


# PipelineModule partition_method doesn't support uneven partitioning
# This allow for loading more layers into selected GPU
# For example if you have 2 gpus - one with 16GB and other with 24GB normal partitioning would throw OOM
# With this implementation you can set partition_split in config so that less layers is loaded onto 16GB GPU
class ManualPipelineModule(PipelineModule):
    def __init__(self, *args, manual_partition_split=None, **kwargs):
        self.manual_partition_split = manual_partition_split
        super().__init__(*args, **kwargs)

    def _partition_layers(self, method='uniform'):
        if method.lower() == 'manual' and self.manual_partition_split is not None:
            num_stages = self._topo.get_dim('pipe')
            stage_id = self._topo.get_coord(self.global_rank).pipe
            num_partitions = len(self.manual_partition_split)
            assert num_partitions == num_stages - 1, f'partition_split must be length {num_stages-1} (pipeline_stages-1), was actually {num_partitions}'

            total_layers = len(self._layer_specs)
            boundaries = [0] + self.manual_partition_split + [total_layers]
            self.parts = boundaries

            # Print some information on the partitioning.
            if self.global_rank == 0:
                for stage in range(num_stages):
                    start = self.parts[stage]
                    stop = self.parts[stage + 1]
                    print(f'stage={stage} layers={stop - start}')
                    for idx, layer in enumerate(self._layer_specs[start:stop]):
                        name = str(layer)
                        if isinstance(layer, LayerSpec):
                            name = layer.typename.__name__
                        if isinstance(layer, nn.Module):
                            name = layer.__class__.__name__
                        else:
                            try:
                                name = layer.__name__
                            except AttributeError:
                                pass
                        print(f'    {idx+start:2d}: {name}')
                if self.loss_fn:
                    try:
                        print(f'  loss: {self.loss_fn.__name__}')
                    except AttributeError:
                        print(f'  loss: {self.loss_fn.__class__.__name__}')

            self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id+1])
        else:
            super()._partition_layers(method)
