import sqlite3
from pathlib import Path
import os
import io
from collections import defaultdict

import torch


class Cache:
    def __init__(self, path: str, fingerprint: str, shard_size_gb=1):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.metadata_db = self.path / 'metadata.db'
        self.shard_size_gb = shard_size_gb
        os.makedirs(self.path, exist_ok=True)

        self.init()


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        assert isinstance(idx, int)
        shard_id, shard_index = self.items[idx]
        offset, size = self.shard_metadata[shard_id][shard_index]
        f = self.open_files.setdefault(shard_id, open(self.path / f'shard_{shard_id}.bin', 'rb'))
        f.seek(offset)
        byte_string = f.read(size)
        buffer = io.BytesIO(byte_string)
        item = torch.load(buffer, map_location='cpu')
        return item


    def init(self):
        print('[CACHE] Initializing')
        # create database
        self.con = sqlite3.connect(self.metadata_db, autocommit=False)

        # check fingerprint, clear cache if different
        self.con.execute('CREATE TABLE IF NOT EXISTS fingerprint(value)')
        existing_fingerprint = self.con.execute('SELECT value FROM fingerprint').fetchone()
        if existing_fingerprint is not None:
            existing_fingerprint = existing_fingerprint[0]
            print(f'[CACHE] Existing cache has fingerprint {existing_fingerprint}')
            if self.fingerprint != existing_fingerprint:
                print('[CACHE] Fingerprint changed, deleting existing cache files')
                self.clear()
                return
        else:
            print(f'[CACHE] Storing new fingerprint: {self.fingerprint}')
            self.con.execute('INSERT INTO fingerprint VALUES(?)', (self.fingerprint,))

        # items table, current length, next shard index
        self.con.execute('CREATE TABLE IF NOT EXISTS items(shard, shard_index)')
        self.items = self.con.execute('SELECT shard, shard_index FROM items').fetchall() or []
        max_existing_shard = -1
        for shard, _ in self.items:
            max_existing_shard = max(max_existing_shard, shard)
        self.shard = max_existing_shard + 1  # current shard to write to
        self.shard_file = None
        print(f'[CACHE] Existing cache length: {len(self)}')

        # shard metadata
        self.shard_metadata = defaultdict(list)
        for table_name, in self.con.execute('SELECT name FROM sqlite_master').fetchall():
            if table_name.startswith('shard_'):
                shard_id = int(table_name.split('_')[-1])
                for entry in self.con.execute(f'SELECT offset, size FROM {table_name}').fetchall():
                    self.shard_metadata[shard_id].append(entry)
        self.open_files = {}

        # commit
        self.con.commit()


    def clear(self):
        '''Deletes all cache files from disk. Calls init() again.'''
        self.con.close()
        os.remove(self.metadata_db)
        for bin_path in self.path.glob('*.bin'):
            os.remove(bin_path)
        self.init()


    def create_new_shard(self):
        self.shard_file = open(self.path / f'shard_{self.shard}.bin', 'wb')
        self.shard_table = f'shard_{self.shard}'
        print(f'[CACHE] Creating new shard: {self.shard_table}')
        self.con.execute(f'CREATE TABLE {self.shard_table}(offset, size)')
        self.shard_index = 0
        self.offset = 0


    def finalize_current_shard(self):
        if self.shard_file is None:
            # no-op if already finalized
            return
        self.shard_file.close()
        self.shard_file = None
        self.shard += 1
        self.con.commit()


    def add(self, item):
        if self.shard_file is None:
            self.create_new_shard()
        buffer = io.BytesIO()
        torch.save(item, buffer)
        bytes_view = buffer.getbuffer()
        self.shard_file.write(bytes_view)

        # update items metadata
        item = (self.shard, self.shard_index)
        self.items.append(item)
        self.con.execute('INSERT INTO items VALUES(?, ?)', item)
        self.shard_index += 1

        # update shard metadata
        size = len(bytes_view)
        entry = (self.offset, size)
        self.shard_metadata[self.shard].append(entry)
        self.con.execute(f'INSERT INTO {self.shard_table} VALUES (?, ?)', entry)
        self.offset += size

        # create new shard when existing one is large enough
        current_size_gb = self.shard_file.tell() / 1_000_000_000
        if current_size_gb >= self.shard_size_gb:
            self.finalize_current_shard()


# for testing
if __name__ == '__main__':
    cache = Cache('/home/anon/tmp/cache_test', 'foo', shard_size_gb=0.001)

    tensor = torch.zeros((100_000,))
    for _ in range(10):
        cache.add({'key1': tensor})
    cache.finalize_current_shard()

    print(cache[0])
    print(cache[1])
