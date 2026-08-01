# Historical-folder retention decision

Decision: `RETAIN_READ_ONLY_PENDING_BACKUP_AND_UNIQUENESS_VERIFICATION`.

`../gnn-fraud-old` is treated as a historical archive and was not modified. It is not required by the CoreGraph `.git` pointer, curated repository metadata, normal path configuration, or runtime code. However, the build cannot truthfully verify that a separate backup opens, contains every expected top-level folder, or preserves every unique uncommitted file. Canonical prediction archives are also not recovered elsewhere in this session.

Deletion is therefore not authorised. Canonical evidence recovery and active-repository Git independence now pass, but a future archival pass must still verify the user-managed backup and unique-file inventory before offering a recoverable move-to-trash operation.
