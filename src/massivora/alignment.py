import os
import re
import time
import logging
import sys
from massivora import compute
import zarr
import pickle
import subprocess
import shutil

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio import ExPASy
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

class TextAlignment(MultipleSeqAlignment):
    def __init__(self, records):
        # Allowing contruction from file path
        if type(records) == str:
            if os.path.isfile(records):
                logger.info(f'Loading alignment from file: {records}')
                _, ext = os.path.splitext(records)
                self.alignmentName = os.path.basename(_)
                if ext.lower() == ".sto":
                    fmt = "stockholm"
                elif ext.lower().startswith(".fa") or ext.lower() == ".a2m":
                    fmt = "fasta"
                else:
                    logger.error(f'Input file {os.path.basename(_)} has an unsupported format {ext}')
                    raise NotImplementedError
                records = AlignIO.read(records, fmt)
            else: raise FileNotFoundError(f"File not found: {records}")
        elif isinstance(records, SeqRecord):
            # construct with single record
            records = [records]
        # construct with biopython supported iterable
        super().__init__(records)
        try:
            self.querySequence = self._records[0]
            self._update_info()
        except IndexError:
            logger.error("Alignment is empty")
            raise ValueError("Alignment is empty")
        self.saved_columns = []
        logger.debug(f'Alignment loaded: {len(self)} sequences, {len(self[0])} positions')

    def _parse_description(self, description):
        # TODO: Try the taxid first, if not exist, try the species name
        pattern = r"OS=(.*?)(?=\s+(?:OX|GN|PE|SV)=|$)"\
                + r"(?:\sOX=(.*?))?(?=\s+(?:GN|PE|SV)=|$)"\
                + r"(?:\sGN=(.*?))?(?=\s+(?:PE|SV)=|$)"\
                + r"(?:\sPE=(.*?))?(?=\s+(?:SV)=|$)"\
                + r"(?:\sSV=(.*?))?$"
        #r"OS=(.*?)(?=\s+(?:GN|PE|SV)=|$)(?:\s(?:GN)=(.*?))?(?=\s+(?:PE|SV)=|$)(?:\s(?:PE)=(.*?))?(?=\s+(?:SV)=|$)(?:\s(?:SV)=(.*?))?$"
        match = re.search(pattern, description)
        if match:
            # matching pattern OS=... OX=... GN=... PE=... SV=...
            return match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
        return None, None, None, None, None

    def _analyze_descriptions(self):
        self._desc_list = []
        # TODO: try the difference between dict[list] and list[list]
        for record in self._records:
            os_, ox, gn, pe, sv = self._parse_description(record.description)
            self._desc_list.append({
                "id": record.id,
                "OS": os_,
                "OX": int(ox) if ox else None,
                "GN": gn,
                "PE": pe,
                "SV": sv
            })

    def _update_info(self):
        self.sequenceLength = len(self._records[0].seq)
        self.sequenceCount = len(self._records)
        self.matrix = self.Get_Numpy_Array()
        self.encoded_matrix = self.Get_Numpy_Array(mapped=True)

    @classmethod
    def From_SwissProt(cls, protein_id):
        try:
            handle = ExPASy.get_sprot_raw(protein_id)
            record = SeqIO.read(handle, 'swiss')
            return cls(record)
        except ValueError as e:
            logger.error(f"Failed to fetch protein {protein_id}: {e}")
            return None

    def AnalogueSearch(self, query=None, threshold=0.2, **search_args):
        if not query:
            _id = self._records[0].id
            query = os.path.join(search_args['working_dir'], f'{_id}.fasta')
            SeqIO.write(self._records[0], query, "fasta")
        search_threshold = str(threshold * self.sequenceLength)
        cmd = [
            search_args['binary'],
            "-N", str(search_args['iterations']),
            "-o", '/dev/null',
            "-A", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.sto'),
            "--tblout", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.tblout'),
            "--domtblout", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.domtblout'),
            "--noali",
            "--notextw",
            "-T", search_threshold,
            "--domT", search_threshold,
            "--incT", search_threshold,
            "--incdomT", search_threshold,
            "--cpu", str(search_args['cpu']),
            query, search_args['database']
        ]
        logger.debug("Running Jackhmmer with the following command line: "+ ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error(f"Jackhmmer failed with return code {r.returncode}")
            raise RuntimeError(f"Jackhmmer failed with return code {r.returncode}")
        else:
            self.__init__(os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.sto'))

    def Get_Numpy_Array(self, mapped=False, string=None):
        if not string: string = ''.join([str(record.seq) for record in self])
        matrix = compute.getAlignmentInNumpy(string, self.sequenceLength, self.sequenceCount)
        # First change all the '.' to '-'
        matrix[matrix == b'.'] = b'-'
        if mapped:
            # Then map the characters to numbers
            # All the invalid characters are mapped to 21
            mapping = np.full(128, 21, dtype=np.int8)
            valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
            values = np.arange(-20, 21, dtype=np.int8)
            for char, value in zip(valid_chars, values): mapping[ord(char)] = value
            ascii_codes = np.frombuffer(matrix.tobytes(), dtype=np.int8)
            mapped_flat = mapping[ascii_codes]
            matrix = mapped_flat.reshape(matrix.shape)
        return matrix

    def Filtering_MSA_Invalid(self):
        # Remove sequences with invalid characters from the alignment
        valid_rows = np.where(np.max(self.encoded_matrix, axis=1) <= 20)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), valid_rows)
 
        if not hasattr(self, "_desc_list"):
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
        else:
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
                del self._desc_list[i]
        self._update_info()
        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Filtering_MSA_Gap(self, row_keep_percentage, col_keep_percentage):
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        
        # Remove gaps if the input is a stockholm file
        columns_to_keep = np.array([char != '-' for char in self._records[0].seq])
        matrix = self.matrix.copy()
        encoded_matrix = self.encoded_matrix.copy()
        encoded_matrix = encoded_matrix[:, columns_to_keep]

        # Calculate the ratio of '-' in each row and filter out rows with a ratio higher than row_keep_percentage
        logger.debug("Calculate rows to keep")
        row_gap_ratios = np.sum(encoded_matrix == 0, axis=1) / encoded_matrix.shape[1]
        indices_to_keep = np.where(row_gap_ratios <= row_keep_percentage)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), indices_to_keep)
        encoded_matrix = encoded_matrix[indices_to_keep]
        logger.debug(f"Matrix shape after row filtering: {encoded_matrix.shape}")

        # Calculate the ratio of '-' in each column and filter out columns with a ratio higher than col_keep_percentage
        logger.debug("Calculate columns to keep")
        L = encoded_matrix.shape[0]
        col_gap_ratios = np.sum(encoded_matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < col_keep_percentage)[0]
        self.saved_columns = cols_to_keep.tolist() # Just keep it as a mask

        # Calculate identity to query for remaining sequences
        temp_matrix = encoded_matrix.copy()[:, cols_to_keep]
        query_seq = temp_matrix[0]
        ident_to_query = np.round(
            (
                np.sum(temp_matrix == query_seq.reshape(1, -1), axis=1)
                + encoded_matrix.shape[1] # removed columns are treated as matches
                - temp_matrix.shape[1]
            ) / encoded_matrix.shape[1]
        , 3)
        for i, identity in zip(indices_to_keep, ident_to_query):
            self._desc_list[i]["query_identity"] = float(identity)

        logger.debug("Reconstructing alignment")
        matrix = matrix[:, columns_to_keep]
        matrix = matrix[indices_to_keep]
        # matrix = matrix[:, cols_to_keep]
        # matrix = np.char.upper(matrix)
        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]
        assert len(self._records) == L
        for i, record in enumerate(self):
            record.letter_annotations = {}
            byte_t = bytes()
            for j in matrix[i]: byte_t += j
            record.seq = Seq(byte_t.decode())

        self._update_info()

        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Best_Reciprocal_Hit(self, paralog_threshold=0.9, allowed_error=0.02):
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        encoded_matrix = self.encoded_matrix.copy()
        if self._desc_list[0].get("query_identity") is None:
            if self.saved_columns:
                reduced_matrix = encoded_matrix[:, self.saved_columns]
                query_seq = reduced_matrix[0]
                ident_to_query = np.round(np.sum(reduced_matrix == query_seq.reshape(1, -1), axis=1)
                                          + self.sequenceLength
                                          - len(self.saved_columns), 3) / self.sequenceLength
            else:
                query_seq = encoded_matrix[0]
                ident_to_query = np.round(np.sum(encoded_matrix == query_seq.reshape(1, -1), axis=1), 3) / self.sequenceLength
            for i, identity in enumerate(ident_to_query):
                self._desc_list[i]["query_identity"] = float(identity)

        desc_df = pd.DataFrame(self._desc_list)
        indices_to_keep1 = (
            desc_df
            .sort_values(by="query_identity", ascending=False)   # highest -> lowest
            .groupby("OX", sort=False)
            .head(1)                            # keep first (highest identity) per OX
            .index.to_numpy()                   # these are the row indices in self._records
        )
        logger.debug("This alignment has {} unique species.".format(len(indices_to_keep1)))

        # Get query taxid
        query_name = self[0].name
        paralog_taxid = None
        for i, rec in enumerate(self[1:]):
            if query_name in rec.name:
                paralog_taxid = self._desc_list[i+1]["OX"]
                break
        if paralog_taxid is None:
            logger.warning("No paralog taxid could be identified; skipping paralog filtering.")
            return

        # Collect all paralog records sharing this taxid
        paralog_indices = []
        for i, desc in enumerate(self._desc_list):
            if desc["OX"] == paralog_taxid and self._desc_list[i]["query_identity"] < paralog_threshold:
                paralog_indices.append(i)
        if len(paralog_indices) == 0:
            logger.warning(f"No paralog sequences found for identified taxid {paralog_taxid}; skipping paralog filtering.")
            return

        # Calculate identity of each sequence to the paralog group
        if self.saved_columns:
            reduced_matrix = encoded_matrix[:, self.saved_columns]
            paralog_seqs = reduced_matrix[paralog_indices]
            reduced_matrix = reduced_matrix[indices_to_keep1, :]
            ident_to_paralog = np.round(np.sum(reduced_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2)
                                        + self.sequenceLength - len(self.saved_columns), 3) / self.sequenceLength
        else:
            paralog_seqs = encoded_matrix[paralog_indices]
            ident_to_paralog = np.round(np.sum(encoded_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2), 3) / self.sequenceLength

        # Compare similarities, if sequence is more similar to any paralog than the query, remove it
        indices_to_keep2 = [0]  # always keep query
        for i, i_all in enumerate(indices_to_keep1):
            max_paralog_id = np.max(ident_to_paralog[:, i])
            if max_paralog_id < self._desc_list[i_all]["query_identity"] + allowed_error:
                indices_to_keep2.append(i_all)
        
        all_indices = np.arange(1, len(self._records))
        indices_to_delete = np.setdiff1d(all_indices, indices_to_keep2)
        logger.debug("Best reciprocal hits retained. The alignment now has {} sequences.".format(len(indices_to_keep2)))

        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]

        self._update_info()

    def Filtering_MSA_Paralog(self):
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        if self._desc_list[0].get("query_identity") is None:
            matrix = self.matrix.copy()
            query_seq = matrix[0]
            ident_to_query = np.sum(matrix == query_seq.reshape(1, -1), axis=1) / self.sequenceLength
            for i, identity in enumerate(ident_to_query):
                self._desc_list[i]["query_identity"] = float(identity)
        
        desc_df = pd.DataFrame(self._desc_list)
        indices_to_keep = (
            desc_df
            .sort_values(by="query_identity")   # lowest -> highest
            .groupby("OX", sort=False)
            .tail(1)                            # keep last (highest identity) per OX
            .index.to_numpy()                   # these are the row indices in self._records
        )
        all_indices = np.arange(1, len(self._records))
        indices_to_delete = np.setdiff1d(all_indices, indices_to_keep)
        logger.debug("Paralogs of query removed. The alignment now has {} sequences.".format(len(indices_to_keep)))

        if not hasattr(self, "_desc_list"):
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
        else:
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
                del self._desc_list[i]

    def Filtering_By_Query_Paralog_Similarity(self, paralog_threshold=0.9):
        """Remove sequences that are more similar to the paralog group than to the query.

        The first record (index 0) is assumed to be the query. Paralogs are
        identified in three steps:
          1) Use the query record's name to find other records with the same
             name in the alignment.
          2) For the first such hit (non-query), parse its description to
             extract the taxid (OX field).
          3) Use this taxid to collect all records in the alignment that share
             this taxid; these form the paralog group.

        Any non-paralog sequence whose sequence identity to the *representative*
        paralog (the paralog with highest sequence identity to the query) is
        greater than its identity to the query by more than the given threshold
        is removed. Query and all paralogs are always kept.

        Parameters
        ----------
        paralog_threshold : float, optional
            Minimal difference (paralog_id - query_id) above which a sequence
            is considered closer to the paralog and therefore removed.
            Default is 0.0 (strictly closer to paralog).
        """
        query_name = self[0].name
        paralog_taxid = None

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()

        for i, rec in enumerate(self[1:]):
            if query_name in rec.name:
                paralog_taxid = self._desc_list[i+1]["OX"]
                break
        if paralog_taxid is None:
            logger.warning("No paralog taxid could be identified; skipping paralog filtering.")
            return

        # Compute sequence identity of every sequence to the query
        matrix = self.matrix.copy()
        if self._desc_list[0].get("query_identity") is None:
            query_seq = matrix[0]
            ident_to_query = np.sum(matrix == query_seq.reshape(1, -1), axis=1) / self.sequenceLength
            for i, identity in enumerate(ident_to_query):
                self._desc_list[i]["query_identity"] = float(identity)

        # Collect all paralog records sharing this taxid
        paralog_indices = []
        for i, desc in enumerate(self._desc_list):
            if desc["OX"] == paralog_taxid and self._desc_list[i]["query_identity"] < paralog_threshold:
                paralog_indices.append(i)
        if len(paralog_indices) == 0:
            logger.warning("No paralog sequences found for identified taxid; skipping paralog filtering.")
            return

        # Calculate identity of each sequence to the paralog group
        paralog_seqs = matrix[paralog_indices]
        ident_to_paralog = np.sum(matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2) / self.sequenceLength

        # Compare similarities, if sequence is more similar to any paralog than the query, remove it
        indices_to_keep = [0]  # always keep query
        for i in range(1, len(self)):
            max_paralog_id = np.max(ident_to_paralog[:, i])
            if max_paralog_id < self._desc_list[i]["query_identity"]+0.02:
                indices_to_keep.append(i)

        # Delete all other sequences
        indices_to_delete = np.setdiff1d(np.arange(len(self)), indices_to_keep)
        if not hasattr(self, "_desc_list"):
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
        else:
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
                del self._desc_list[i]

        self._update_info()
        logger.info(
            f'Filtered alignment contains: {len(self)} sequences after paralog filtering; '
            f'paralog taxid={paralog_taxid}, total paralogs={len(paralog_indices)}'
        )

    def __add__(self, other):
        if not isinstance(other, MultipleSeqAlignment):
            raise NotImplementedError("Only MultipleSeqAlignment can be concatenated")
        
        # Create species-based dictionary for both alignments
        self_species = {}
        other_species = {}

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        if not hasattr(other, "_desc_list"):
            other._analyze_descriptions()

        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in self_species:
                self_species[taxid] = []
            self_species[taxid].append((rec, desc))

        for rec, desc in zip(other[1:], other._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in other_species:
                other_species[taxid] = []
            other_species[taxid].append((rec, desc))

        logger.debug(f"Found {len(self_species)} species in alignment 1")
        logger.debug(f"Found {len(other_species)} species in alignment 2")

        # Merge records from both alignments
        merged = [SeqRecord(self[0].seq + other[0].seq, id='Query', name='Query', description=self[0].id+'_'+other[0].id)]
        common_species = set(self_species.keys()) & set(other_species.keys())
        logger.debug(f"Found {len(common_species)} common species")
        redundant_species = []
        ## First concatenate the sequences that are unique to each alignment
        logger.debug("Looping through common species")
        for species in common_species:
            if len(self_species[species]) == 1 and len(other_species[species]) == 1:
                rec1 = self_species[species][0][0]
                rec2 = other_species[species][0][0]
                # logger.debug(f'rec1.name: {rec1.name}, rec2.name: {rec2.name}, species: {species}')
                new_rec = SeqRecord(rec1.seq + rec2.seq, id=str(species), name=str(species), description=rec1.id+'_'+rec2.id)
                merged.append(new_rec)
            else:
                redundant_species.append(species)
        ## Then concatenate the sequences that are redundant in both alignments
        logger.debug(f"Found {len(redundant_species)} redundant species")
        for species in redundant_species:
            identity = 0
            for rec, desc in self_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_self = rec
            identity = 0
            for rec, desc in other_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_other = rec
            new_rec = SeqRecord(most_similar_seq_in_self.seq + most_similar_seq_in_other.seq, id=str(species), name=str(species), description=most_similar_seq_in_self.id+'_'+most_similar_seq_in_other.id)
            merged.append(new_rec)
        logger.info(f'Alignment merged: {len(merged)} sequences, {len(merged[0])} positions')
        return TextAlignment(merged)

    def Downsample_Randomly(self, to):
        if to >= self.sequenceCount:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.sequenceCount} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.sequenceCount), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)
        chosen_records = [self._records[i] for i in chosen_indices]
        super().__init__(chosen_records)
        self._update_info()

    def To_Zarr(self, filename=None, overwrite=False):
        if not filename: filename = f'{self.alignmentName}'
        if os.path.exists(filename) & os.path.exists(os.path.join(filename, 'c')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'c'), ignore_errors=True)
                os.remove(os.path.join(filename, 'zarr.json'))
                os.remove(os.path.join(filename, 'metadata.pkl'))
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        species_list = ['Query']
        species_index_map = [0]

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()

        # Group sequences by species
        species = {}
        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid is None: continue
            if taxid not in species.keys(): species[taxid] = []
            species[taxid].append(str(rec.seq))

        p_seq = 1
        tempstr = str(self[0].seq)
        for species, sequences in species.items():
            sequence_count = len(sequences)
            # for multiple records of the same species,
            # put the one with the highest identity to the query sequence at the first position
            if sequence_count != 1:
                identity = 0
                top_identity_index = 0
                q_arr = np.array(list(self.querySequence.seq))
                for this_index, rec in enumerate(sequences):
                    r_arr = np.array(list(rec))
                    this_identity = np.sum(q_arr == r_arr)
                    if this_identity > identity:
                        identity = this_identity
                        top_identity_index = this_index
                sequences.insert(0, sequences.pop(top_identity_index))
            for seq in sequences: tempstr += seq
            species_index_map.append(p_seq)
            species_list.append(species)
            p_seq += sequence_count
        matrix = self.Get_Numpy_Array(mapped=True, string=tempstr)
        z = zarr.create_array(store=filename,shape=(self.sequenceCount, self.sequenceLength), dtype='int8')
        metadata = {
            'species_list': species_list, 
            'species_index_map': species_index_map,
            'saved_columns': list(self.saved_columns)
        }
        with open(os.path.join(filename, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        z[:] = matrix


class ZarrAlignment(object):
    def __init__(self, filename=None):
        self.matrix = np.zeros((0, 0), dtype=np.int8)
        self.metadata = {}
        self.species_list_map = {}
        self.saved_columns = []

        if isinstance(filename, str):
            try:
                with open(os.path.join(filename, 'metadata.pkl'), 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                logger.error(f"Metadata file not found in {filename}, {e}")
                raise FileNotFoundError(f"Metadata file not found in {filename}")
            try:
                arr = zarr.open(filename, mode='r')
                self.matrix = arr[:]
                self.weights = arr.attrs.get('weights', None)
                self.Beff = arr.attrs.get('Beff', None)
            except Exception as e:
                logger.error(f"Error loading Zarr file {filename}, {e}")
                raise FileNotFoundError(f"Error loading Zarr file {filename}")
            self.species_list_map = {species: i for i, species in enumerate(self.metadata['species_list'])}
            self.saved_columns = self.metadata.get('saved_columns', [])

    def __repr__(self):
        return "<%s instance (%i records of length %i) at %x>" % (
            self.__class__,
            self.matrix.shape[0],
            self.matrix.shape[1],
            id(self),
        )

    def __str__(self):
        return f"Alignment from Zarr with {len(self.metadata['species_list'])} species"

    def __add__(self, other):
        # Ensure that Query is the first species in both alignments
        common_species = sorted(list(set(self.metadata['species_list'][1:]) & set(other.metadata['species_list'][1:])))
        common_species.insert(0, self.metadata['species_list'][0])

        new_instance = ZarrAlignment()
        self_saved = list(range(self.matrix.shape[1]))
        other_saved = list(range(other.matrix.shape[1]))
        saved_columns = self_saved + other_saved
        new_instance.metadata = {
            'species_list': common_species,
            'species_index_map': list(range(len(common_species))),
            'saved_columns': saved_columns
        }
        new_instance.saved_columns = new_instance.metadata['saved_columns']
        new_instance.species_list_map = {species: i for i, species in enumerate(new_instance.metadata['species_list'])}

        # species_list_map[species] is the index of the species in the species_list
        # metadata['species_index_map'][species_list_map[species]] is the index of the species in the matrix
        indices1 = np.array([self.metadata['species_index_map'][self.species_list_map[species]] for species in common_species])
        indices2 = np.array([other.metadata['species_index_map'][other.species_list_map[species]] for species in common_species])
        new_instance.matrix = np.concatenate((self.matrix[indices1], other.matrix[indices2]), axis=1)

        return new_instance

    def Column_Gap_Control(self, gap_percentage=50):
        gap_percentage /= 100
        L = self.matrix.shape[0]
        col_gap_ratios = np.sum(self.matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < gap_percentage)[0]
        self.matrix = self.matrix[:, cols_to_keep]
        self.saved_columns = [self.saved_columns[i] for i in cols_to_keep]

    def Downsample_Randomly(self, to):
        if to >= self.matrix.shape[0]:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.matrix.shape[0]} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.matrix.shape[0]), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)  # Ensure the first row is included
        self.matrix = self.matrix[chosen_indices]

        # Update metadata to reflect the downsampled alignment
        kept_count = len(chosen_indices)
        new_species_list = ['Query']
        new_species_index_map = list(range(kept_count))

        for i in range(kept_count-1):
            new_species_list.append(self.metadata['species_list'][chosen_indices[i + 1]])

        self.metadata['species_list'] = new_species_list
        self.metadata['species_index_map'] = new_species_index_map
        self.species_list_map = {species: i for i, species in enumerate(new_species_list)}

    def Reweight_Sequence(self, x=0.8):
        B = self.matrix.shape[0]
        N = self.matrix.shape[1]
        identical_threshold = x * N
        simM = np.full((B,B), N, dtype=np.int16)

        # TODO: parallelize this (or not?)
        for b in range(B-1):
            identical_positions = np.equal(self.matrix[b+1:], self.matrix[b])
            identity_scores = np.sum(identical_positions, axis=1)
            simM[b, b+1:] = identity_scores
            simM[b+1:, b] = identity_scores
        m = np.sum(simM >= identical_threshold, axis=0)
        w = 1/m
        Beff = w.sum()

        self.weights = w.tolist()
        self.Beff = float(Beff)

    def To_Text(self, restore_gaps=True, filename=None):
        # Create reverse mapping from numbers back to characters
        valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
        values = np.arange(-20, 21, dtype=np.int8)
        reverse_mapping = np.full(128, ord('X'), dtype=np.uint8)
        for char, value in zip(valid_chars, values):
            reverse_mapping[value + 20] = ord(char)

        # Convert numeric matrix back to character matrix
        ascii_codes = reverse_mapping[self.matrix + 20]
        char_matrix = np.frombuffer(ascii_codes.tobytes(), dtype='S1').reshape(self.matrix.shape)
        if restore_gaps:
            diff = np.diff(self.saved_columns)
            decrease_indices = np.where(diff < 0)[0]
            monomer_starts = [0]
            for ending in decrease_indices:
                monomer_starts.append(ending + 1)
            monomer_matrices = []
            for i in range(len(monomer_starts)):
                start_idx = monomer_starts[i]
                end_idx = monomer_starts[i + 1] if i + 1 < len(monomer_starts) else len(self.saved_columns)
                monomer_saved_cols = self.saved_columns[start_idx:end_idx]
                monomer_char_matrix = char_matrix[:, start_idx:end_idx]
                monomer_original_length = max(monomer_saved_cols) + 1
                monomer_original_matrix = np.full((self.matrix.shape[0], monomer_original_length), b'-', dtype='S1')
                monomer_original_matrix[:, monomer_saved_cols] = monomer_char_matrix
                monomer_matrices.append(monomer_original_matrix)
            original_matrix = np.concatenate(monomer_matrices, axis=1)
        else:
            original_matrix = char_matrix.copy()
        original_matrix = original_matrix.astype('U1')

        # Convert matrix back to SeqRecord objects
        seq_records = []
        i = 0
        for species, starting_idx in zip(self.metadata['species_list'][:-1], self.metadata['species_index_map'][1:]):
            for _ in range(i, starting_idx):
                row = original_matrix[i]
                sequence_str = ''.join(row.tolist())
                seq_record = SeqRecord(
                    Seq(sequence_str),
                    id=str(i),
                    name=str(i) + ' OX=' + str(species),
                    description='OX=' + str(species)
                )
                seq_records.append(seq_record)
                i += 1
        species = self.metadata['species_list'][-1]
        for _ in range(i, original_matrix.shape[0]):
            row = original_matrix[i]
            sequence_str = ''.join(row.tolist())
            seq_record = SeqRecord(
                Seq(sequence_str),
                id=str(i),
                name=str(i) + ' OX=' + str(species),
                description='OX=' + str(species)
            )
            seq_records.append(seq_record)
            i += 1

        if filename:
            SeqIO.write(seq_records, filename, "fasta")

        return TextAlignment(seq_records)

    def To_Zarr(self, filename=None, overwrite=False):
        if not filename: filename = 'merged_alignment'
        if os.path.exists(filename):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                if os.path.exists(os.path.join(filename, 'c')):
                    shutil.rmtree(os.path.join(filename, 'c'), ignore_errors=True)
                os.remove(os.path.join(filename, 'zarr.json'))
                os.remove(os.path.join(filename, 'metadata.pkl'))
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        z = zarr.create_array(store=filename,shape=self.matrix.shape, dtype='int8')
        z.attrs.update({
            'weights': getattr(self, 'weights', None),
            'Beff': getattr(self, 'Beff', None)
        })
        # TODO: move python pkl metadata to zarr metadata
        self.metadata['saved_columns'] = self.saved_columns
        with open(os.path.join(filename, 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
        z[:] = self.matrix

    def To_Npy(self, filename=None):
        if filename:
            np.save(filename, self.matrix)
        return self.matrix

if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s - From %(name)s: %(message)s'
    logging.basicConfig(format=FORMAT, stream=sys.stderr, level=logging.DEBUG)

    start_time = time.time()
    alignment1 = TextAlignment("/Users/simon/research/EVcouplings/residue_distance/ATPA_HUMAN_b0.2.sto")
    # alignment1 = TextAlignment("/Users/simon/Downloads/RL22_ECOLI_1-110_b0.2/align/RL22_ECOLI_1-110_b0.2.sto")
    alignment1.Filtering_MSA_Gap(50, 50)
    # alignment1.Filtering_MSA_Paralog()
    # df = pd.DataFrame(alignment1._desc_list)
    # df.to_pickle("/Users/simon/Downloads/RL18New.pkl")
    # alignment1.Filtering_By_Query_Paralog_Similarity()
    alignment1.Filtering_MSA_Invalid()
    alignment1.Best_Reciprocal_Hit()
    # alignment1.Filtering_MSA_Gap(100, 50)
    alignment1.To_Zarr("/Users/simon/Downloads/ATPA_HUMAN", overwrite=True)
    alignment2 = TextAlignment("/Users/simon/research/EVcouplings/residue_distance/ATPB_HUMAN_b0.2.sto")
    # alignment2 = TextAlignment("/Users/simon/Downloads/RL31_ECOLI_1-70_b0.2/align/RL31_ECOLI_1-70_b0.2.sto")
    alignment2.Filtering_MSA_Gap(50, 50)
    # alignment2.Filtering_MSA_Paralog()
    # df = pd.DataFrame(alignment2._desc_list)
    # df.to_pickle("/Users/simon/Downloads/RL31New.pkl")
    # alignment2.Filtering_By_Query_Paralog_Similarity()
    alignment2.Filtering_MSA_Invalid()
    alignment2.Best_Reciprocal_Hit()
    # alignment2.Filtering_MSA_Gap(100, 50)
    alignment2.To_Zarr("/Users/simon/Downloads/ATPB_HUMAN", overwrite=True)
    logger.info(f"Time taken for reducing: {time.time() - start_time:.2f}s")
    start_time = time.time()
    zarr1 = ZarrAlignment("/Users/simon/Downloads/ATPA_HUMAN")
    zarr2 = ZarrAlignment("/Users/simon/Downloads/ATPB_HUMAN")
    merged_alignment = zarr1 + zarr2
    merged_alignment.Reweight_Sequence(x=0.8)
    # merged_alignment.Filtering_MSA_Gap(100, 50)
    # merged_alignment.Filtering_MSA_Invalid()
    merged_alignment.To_Zarr("/Users/simon/Downloads/ATPA_ATPB_HUMAN", overwrite=True)
    logger.info(f"Time taken for concatenating: {time.time() - start_time:.2f}s")


