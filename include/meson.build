libfip_inc = include_directories('.')

subdir('fip')

install_subdir('fip',
    install_dir:     get_option('includedir'),
    exclude_files:   ['meson.build', 'fip/meson.build'],
)
